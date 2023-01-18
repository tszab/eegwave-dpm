import torch
import torch as th
import torch.nn.functional as F
import numpy as np

from diffusion_utils.losses import normal_kl
from diffusion_utils.utils import mean_flat

import functorch


def _log1mexp(x):
    """Accurate computation of log(1 - exp(-x)) for x > 0."""
    # From James Townsend's PixelCNN++ code
    # Method from
    # https://cran.r-project.org/web/packages/Rmpfr/vignettes/_log1mexp-note.pdf
    return th.where(x > np.log(2.), th.log1p(-th.exp(-x)), th.log(-th.expm1(-x)))


def _broadcast_from_left(x, shape):
    assert len(shape) >= x.ndim
    return th.broadcast_to(
        x.reshape(x.shape + (1,) * (len(shape) - x.ndim)),
        shape)


_log_sigmoid = th.nn.functional.logsigmoid


### Basic diffusion process utilities


def diffusion_reverse(*, x, z_t, logsnr_s, logsnr_t, x_logvar):
    """q(z_s | z_t, x) (requires logsnr_s > logsnr_t (i.e. s < t))."""
    alpha_st = th.sqrt((1. + th.exp(-logsnr_t)) / (1. + th.exp(-logsnr_s)))
    alpha_s = th.sqrt(th.sigmoid(logsnr_s))
    r = th.exp(logsnr_t - logsnr_s)  # SNR(t)/SNR(s)
    one_minus_r = -th.expm1(logsnr_t - logsnr_s)  # 1-SNR(t)/SNR(s)
    log_one_minus_r = _log1mexp(logsnr_s - logsnr_t)  # log(1-SNR(t)/SNR(s))

    mean = r * alpha_st * z_t + one_minus_r * alpha_s * x

    if isinstance(x_logvar, str):
        if x_logvar == 'small':
            # same as setting x_logvar to -infinity
            var = one_minus_r * th.sigmoid(-logsnr_s)
            logvar = log_one_minus_r + _log_sigmoid(-logsnr_s)
        elif x_logvar == 'large':
            # same as setting x_logvar to nn._log_sigmoid(-logsnr_t)
            var = one_minus_r * th.sigmoid(-logsnr_t)
            logvar = log_one_minus_r + _log_sigmoid(-logsnr_t)
        elif x_logvar.startswith('medium:'):
            _, frac = x_logvar.split(':')
            frac = float(frac)
            assert 0 <= frac <= 1
            min_logvar = log_one_minus_r + _log_sigmoid(-logsnr_s)
            max_logvar = log_one_minus_r + _log_sigmoid(-logsnr_t)
            logvar = frac * max_logvar + (1 - frac) * min_logvar
            var = th.exp(logvar)
        else:
            raise NotImplementedError(x_logvar)
    else:
        assert isinstance(x_logvar, th.Tensor) or isinstance(
            x_logvar, np.ndarray)
        assert x_logvar.shape == x.shape
        # start with "small" variance
        var = one_minus_r * th.sigmoid(-logsnr_s)
        logvar = log_one_minus_r + _log_sigmoid(-logsnr_s)
        # extra variance weight is (one_minus_r*alpha_s)**2
        var += th.square(one_minus_r) * th.sigmoid(logsnr_s) * th.exp(x_logvar)
        logvar = th.logaddexp(
            logvar, 2. * log_one_minus_r + _log_sigmoid(logsnr_s) + x_logvar)
    return {'mean': mean, 'std': th.sqrt(var), 'var': var, 'logvar': logvar}


def diffusion_forward(*, x, logsnr):
    """q(z_t | x)."""
    return {
        'mean': x * th.sqrt(th.sigmoid(logsnr)),
        'std': th.sqrt(th.sigmoid(-logsnr)),
        'var': th.sigmoid(-logsnr),
        'logvar': _log_sigmoid(-logsnr)
    }


def predict_x_from_eps(*, z, eps, logsnr):
    """x = (z - sigma*eps)/alpha."""
    logsnr = _broadcast_from_left(logsnr, z.shape)
    return th.sqrt(1. + th.exp(-logsnr)) * (
            z - eps * th.rsqrt(1. + th.exp(logsnr)))


def predict_xlogvar_from_epslogvar(*, eps_logvar, logsnr):
    """Scale Var[eps] by (1+exp(-logsnr)) / (1+exp(logsnr)) = exp(-logsnr)."""
    return eps_logvar - logsnr


def predict_eps_from_x(*, z, x, logsnr):
    """eps = (z - alpha*x)/sigma."""
    logsnr = _broadcast_from_left(logsnr, z.shape)
    return th.sqrt(1. + th.exp(logsnr)) * (
            z - x * th.rsqrt(1. + th.exp(-logsnr)))


def predict_epslogvar_from_xlogvar(*, x_logvar, logsnr):
    """Scale Var[x] by (1+exp(logsnr)) / (1+exp(-logsnr)) = exp(logsnr)."""
    return x_logvar + logsnr


def predict_x_from_v(*, z, v, logsnr):
    logsnr = _broadcast_from_left(logsnr, z.shape)
    alpha_t = th.sqrt(th.sigmoid(logsnr))
    sigma_t = th.sqrt(th.sigmoid(-logsnr))
    return alpha_t * z - sigma_t * v


def predict_v_from_x_and_eps(*, x, eps, logsnr):
    logsnr = _broadcast_from_left(logsnr, x.shape)
    alpha_t = th.sqrt(th.sigmoid(logsnr))
    sigma_t = th.sqrt(th.sigmoid(-logsnr))
    return alpha_t * eps - sigma_t * x


class ContinuousDiffusion:

    def __init__(self, *, mean_type: str = 'eps', logvar_type: str = 'fixed_large', infer_steps: int = None):
        self.mean_type = mean_type
        self.logvar_type = logvar_type
        self.infer_steps = infer_steps

    def _run_model(self, *, z, logsnr, model_fn, clip_x):
        model_output = model_fn(z, logsnr)
        if self.mean_type == 'eps':
            model_eps = model_output
        elif self.mean_type == 'x':
            model_x = model_output
        elif self.mean_type == 'v':
            model_v = model_output
        elif self.mean_type == 'both':
            _model_x, _model_eps = th.chunk(model_output, 2, dim=1)  # pylint: disable=invalid-name
        else:
            raise NotImplementedError(self.mean_type)

        # get prediction of x at t=0
        if self.mean_type == 'both':
            # reconcile the two predictions
            model_x_eps = predict_x_from_eps(z=z, eps=_model_eps, logsnr=logsnr)
            wx = _broadcast_from_left(th.sigmoid(-logsnr), z.shape)
            model_x = wx * _model_x + (1. - wx) * model_x_eps
        elif self.mean_type == 'eps':
            model_x = predict_x_from_eps(z=z, eps=model_eps, logsnr=logsnr)
        elif self.mean_type == 'v':
            model_x = predict_x_from_v(z=z, v=model_v, logsnr=logsnr)

        # clipping
        if clip_x:
            model_x = th.clip(model_x, -1., 1.)

        # get eps prediction if clipping or if mean_type != eps
        if self.mean_type != 'eps' or clip_x:
            model_eps = predict_eps_from_x(z=z, x=model_x, logsnr=logsnr)

        # get v prediction if clipping or if mean_type != v
        if self.mean_type != 'v' or clip_x:
            model_v = predict_v_from_x_and_eps(
                x=model_x, eps=model_eps, logsnr=logsnr)

        return {'model_x': model_x,
                'model_eps': model_eps,
                'model_v': model_v}

    def predict(self, *, z_t, logsnr_t, logsnr_s, clip_x=None,
                model_output=None, model_fn=None):
        """p(z_s | z_t)."""
        assert logsnr_t.shape == logsnr_s.shape == (z_t.shape[0],)
        if model_output is None:
            assert clip_x is not None
            if model_fn is None:
                raise NotImplementedError
            model_output = self._run_model(
                z=z_t, logsnr=logsnr_t, model_fn=model_fn, clip_x=clip_x)

        logsnr_t = _broadcast_from_left(logsnr_t, z_t.shape)
        logsnr_s = _broadcast_from_left(logsnr_s, z_t.shape)

        pred_x = model_output['model_x']
        if self.logvar_type == 'fixed_small':
            pred_x_logvar = 'small'
        elif self.logvar_type == 'fixed_large':
            pred_x_logvar = 'large'
        elif self.logvar_type.startswith('fixed_medium:'):
            pred_x_logvar = self.logvar_type[len('fixed_'):]
        else:
            raise NotImplementedError(self.logvar_type)

        out = diffusion_reverse(
            z_t=z_t, logsnr_t=logsnr_t, logsnr_s=logsnr_s,
            x=pred_x, x_logvar=pred_x_logvar)
        out['pred_x'] = pred_x
        return out

    def vb(self, *, x, z_t, logsnr_t, logsnr_s, model_output):
        assert x.shape == z_t.shape
        assert logsnr_t.shape == logsnr_s.shape == (z_t.shape[0],)
        q_dist = diffusion_reverse(
            x=x,
            z_t=z_t,
            logsnr_t=_broadcast_from_left(logsnr_t, x.shape),
            logsnr_s=_broadcast_from_left(logsnr_s, x.shape),
            x_logvar='small')
        p_dist = self.predict(
            z_t=z_t, logsnr_t=logsnr_t, logsnr_s=logsnr_s,
            model_output=model_output)
        kl = normal_kl(
            mean1=q_dist['mean'], logvar1=q_dist['logvar'],
            mean2=p_dist['mean'], logvar2=p_dist['logvar'])
        return mean_flat(kl) / np.log(2.)

    def training_losses(self, model_fn, x, logsnr_schedule_fn, num_steps, mean_loss_weight_type, target_model_fn=None):
        assert x.dtype in [th.float32, th.float64]
        assert isinstance(num_steps, int)
        eps = th.randn_like(x)
        bc = lambda z: _broadcast_from_left(z, x.shape)

        # sample logsnr
        if num_steps > 0:
            assert num_steps >= 1
            i = th.randint(num_steps, size=(x.shape[0],), dtype=x.dtype, device=x.device)
            u = (i + 1).type(x.dtype) / num_steps
        else:
            # continuous time
            u = torch.rand(size=(x.shape[0],), dtype=x.dtype, device=x.device)
        logsnr = logsnr_schedule_fn(u)
        assert logsnr.shape == (x.shape[0],)

        # sample z ~ q(z_logsnr | x)
        z_dist = diffusion_forward(x=x, logsnr=bc(logsnr))
        z = z_dist['mean'] + z_dist['std'] * eps

        # get denoising target
        if target_model_fn is not None:  # distillation
            assert num_steps >= 1

            # two forward steps of DDIM from z_t using teacher
            with torch.no_grad():
                teach_out_start = self._run_model(
                    z=z, logsnr=logsnr, model_fn=target_model_fn, clip_x=False)
            x_pred = teach_out_start['model_x']
            eps_pred = teach_out_start['model_eps']

            u_mid = u - 0.5 / num_steps
            logsnr_mid = logsnr_schedule_fn(u_mid)
            stdv_mid = bc(th.sqrt(th.sigmoid(-logsnr_mid)))
            a_mid = bc(th.sqrt(th.sigmoid(logsnr_mid)))
            z_mid = a_mid * x_pred + stdv_mid * eps_pred

            with torch.no_grad():
                teach_out_mid = self._run_model(z=z_mid,
                                                logsnr=logsnr_mid,
                                                model_fn=target_model_fn,
                                                clip_x=False)
            x_pred = teach_out_mid['model_x']
            eps_pred = teach_out_mid['model_eps']

            u_s = u - 1. / num_steps
            logsnr_s = logsnr_schedule_fn(u_s)
            stdv_s = bc(th.sqrt(th.sigmoid(-logsnr_s)))
            a_s = bc(th.sqrt(th.sigmoid(logsnr_s)))
            z_teacher = a_s * x_pred + stdv_s * eps_pred

            # get x-target implied by z_teacher (!= x_pred)
            a_t = bc(th.sqrt(th.sigmoid(logsnr)))
            stdv_frac = bc(th.exp(
                0.5 * (F.softplus(logsnr) - F.softplus(logsnr_s))))
            x_target = (z_teacher - stdv_frac * z) / (a_s - stdv_frac * a_t)
            x_target = th.where(bc(i == 0), x_pred, x_target)
            eps_target = predict_eps_from_x(z=z, x=x_target, logsnr=logsnr)

        else:  # denoise to original data
            x_target = x
            eps_target = eps

        # also get v-target
        v_target = predict_v_from_x_and_eps(
            x=x_target, eps=eps_target, logsnr=logsnr)

        # denoising loss
        model_output = self._run_model(
            z=z, logsnr=logsnr, model_fn=model_fn, clip_x=False)
        x_mse = mean_flat(th.square(model_output['model_x'] - x_target))
        eps_mse = mean_flat(th.square(model_output['model_eps'] - eps_target))
        v_mse = mean_flat(th.square(model_output['model_v'] - v_target))
        if mean_loss_weight_type == 'constant':  # constant weight on x_mse
            loss = x_mse
        elif mean_loss_weight_type == 'snr':  # SNR * x_mse = eps_mse
            loss = eps_mse
        elif mean_loss_weight_type == 'snr_trunc':  # x_mse * max(SNR, 1)
            loss = th.maximum(x_mse, eps_mse)
        elif mean_loss_weight_type == 'v_mse':
            loss = v_mse
        else:
            raise NotImplementedError(mean_loss_weight_type)
        return {'loss': loss}

    def ddim_step(self, model_fn, i, z_t, num_steps, logsnr_schedule_fn, clip_x):
        shape, dtype = z_t.shape, z_t.dtype
        logsnr_t = logsnr_schedule_fn((i + 1.) / num_steps)
        logsnr_s = logsnr_schedule_fn(i / num_steps)
        model_out = self._run_model(
            z=z_t,
            logsnr=th.full((shape[0],), logsnr_t, device=logsnr_t.device),
            model_fn=model_fn,
            clip_x=clip_x)
        x_pred_t = model_out['model_x']
        eps_pred_t = model_out['model_eps']
        stdv_s = th.sqrt(th.sigmoid(-logsnr_s))
        alpha_s = th.sqrt(th.sigmoid(logsnr_s))
        z_s_pred = alpha_s * x_pred_t + stdv_s * eps_pred_t
        return th.where(i == 0, x_pred_t, z_s_pred)

    def bwd_dif_step(self, model_fn, i, z_t, num_steps, logsnr_schedule_fn, clip_x):
        shape, dtype = z_t.shape, z_t.dtype
        logsnr_t = logsnr_schedule_fn((i + 1.) / num_steps)
        logsnr_s = logsnr_schedule_fn(i / num_steps)
        z_s_dist = self.predict(
            z_t=z_t,
            logsnr_t=th.full((shape[0],), logsnr_t, device=logsnr_t.device),
            logsnr_s=th.full((shape[0],), logsnr_s, device=logsnr_t.device),
            clip_x=clip_x,
            model_fn=model_fn
        )
        eps = th.randn_like(z_t)
        return th.where(
            i == 0, z_s_dist['pred_x'], z_s_dist['mean'] + z_s_dist['std'] * eps)

    @th.no_grad()
    def sample_loop(self, model_fn, init_x, logsnr_schedule_fn, sampler, clip_x):
        if sampler == 'ddim':
            body_fun = lambda i, z_t: self.ddim_step(model_fn,
                i, z_t, self.infer_steps, logsnr_schedule_fn, clip_x)
        elif sampler == 'noisy':
            body_fun = lambda i, z_t: self.bwd_dif_step(model_fn,
                i, z_t, self.infer_steps, logsnr_schedule_fn, clip_x)
        else:
            raise NotImplementedError(sampler)

        # loop over t = num_steps-1, ..., 0
        final_x = init_x
        for t in list(range(self.infer_steps))[::-1]:
            t = th.tensor(t, dtype=final_x.dtype, device=final_x.device)
            final_x = body_fun(t, final_x)

        assert final_x.shape == init_x.shape and final_x.dtype == init_x.dtype
        return final_x

