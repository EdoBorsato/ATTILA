"""
One-dimensional Sersic fits of surface-brightness profiles with MCMC.

This module provides the ingredients of a 1D model for a radial
intensity profile (`Sersic`, `Sky`, `IntrinsicScatter`), a simple
uniform prior (`PriorSetter`), the log-probabilities of the two
available models and the classes that sample them with `emcee`:

- `LogProbability_SersicSkyIS` / `FitProfile`: Sersic profile plus a
  constant sky level plus an intrinsic scatter term (5 parameters:
  `I0`, `Re`, `n`, `sky`, `log_f`);
- `LogProbability_SersicIS` / `FitProfile_noSky`: the same model
  without the sky level (4 parameters: `I0`, `Re`, `n`, `log_f`).

It is the last step of the ATTILA flow: the profiles fitted here are
the circularized or major-axis profiles extracted from the isophotes
(`attila.fitting.isophote.IsophotesAnalysis`), which are in turn
measured on the partial residuals of the AstroPhot 2D deblending.
"""
import numpy as np
import matplotlib.pyplot as plt

from astropy.convolution import convolve
import emcee
import corner

class Sersic:
    """
    One-dimensional Sersic profile normalised at the centre.

    The profile is

        I(r) = I0 * exp(-b_n * (r / Re) ** (1 / n))

    where `b_n` is the Sersic constant, which makes `Re` the radius
    enclosing half of the total light. Unlike the PyAutoGalaxy
    convention, where the normalisation is the intensity at the
    effective radius, here the normalisation `I0` is the intensity at
    `r = 0`.

    Attributes
    ----------
    I0 : float
        The central intensity, in the units of the fitted profile.
    Re : float
        The effective (half-light) radius, in the units of the radii
        the profile is evaluated at.
    n : float
        The Sersic index.
    """

    def __init__(self, I0, Re, n):
        """
        Store the parameters of the Sersic profile.

        Parameters
        ----------
        I0 : float
            The central intensity, in the units of the fitted profile.
        Re : float
            The effective (half-light) radius, in the units of the
            radii the profile is evaluated at.
        n : float
            The Sersic index.
        """

        self.I0 = I0
        self.Re = Re
        self.n = n

    @property
    def Sersic_constant(self):
        """
        Return the Sersic constant `b_n` for the current Sersic index.

        The constant ensures that the effective radius contains half
        of the total integrated light of the profile. It is computed
        with the asymptotic series in `1 / n`, truncated at the fourth
        order, with the same expression used by PyAutoGalaxy
        (`autogalaxy.profiles.light.standard.sersic`).

        Returns
        -------
        float
            The Sersic constant `b_n`.
        """

        return (
            (2 * self.n)
            - (1.0 / 3.0)
            + (4.0 / (405.0 * self.n))
            + (46.0 / (25515.0 * self.n**2))
            + (131.0 / (1148175.0 * self.n**3))
            - (2194697.0 / (30690717750.0 * self.n**4))
        )

    def model_from_radius(self, radius):
        """
        Evaluate the Sersic profile at the input radii.

        Parameters
        ----------
        radius : numpy.ndarray
            The radii at which the profile is evaluated, in the same
            units as `Re`.

        Returns
        -------
        numpy.ndarray
            The intensity at each radius, in the units of `I0`, with
            the same shape as `radius`.
        """
        return (self.I0 * np.exp(-self.Sersic_constant * (((radius / self.Re) ** (1.0 / self.n)))))

class Sky:
    """
    Constant sky level added to a one-dimensional profile.

    Attributes
    ----------
    sky : float
        The sky level, in the units of the fitted profile.
    """

    def __init__(self, sky):
        """
        Store the sky level.

        Parameters
        ----------
        sky : float
            The sky level, in the units of the fitted profile.
        """
        self.sky = sky

    def model_from_radius(self, radius):
        """
        Evaluate the constant sky level at the input radii.

        Parameters
        ----------
        radius : numpy.ndarray
            The radii of the profile; only their shape is used.

        Returns
        -------
        numpy.ndarray
            An array with the shape of `radius` filled with `sky`.
        """
        return np.ones(radius.shape) * self.sky

class IntrinsicScatter:
    """
    Intrinsic scatter term of the likelihood.

    The scatter is modelled as a fraction `f` of the model intensity,
    added in quadrature to the measurement errors (see
    `LogProbability_SersicSkyIS.log_likelihood`).

    Attributes
    ----------
    log_f : float
        The natural logarithm of the fractional scatter `f`.
    """

    def __init__(self, log_f):
        """
        Store the logarithm of the fractional intrinsic scatter.

        Parameters
        ----------
        log_f : float
            The natural logarithm of the fractional scatter `f`.
        """
        self.log_f = log_f


# Default uniform priors, one dictionary per parameter in the order of
# the parameter vector of the Sersic + sky model: I0, Re, n, sky,
# log_f. The limits on n (0.36, 8) are the same limits AstroPhot sets
# on the Sersic index.
default_prior_dict_list = [
        {'type': 'uniform',
         'min': 0,
         'max': 200}, # I0
         {'type': 'uniform',
         'min': 0,
         'max': 150}, # Re
         {'type': 'uniform',
         'min': 0.36,
         'max': 8}, # n
         {'type': 'uniform',
         'min': -5,
         'max': 10}, # sky
        {'type': 'uniform',
          'min': -6,
          'max': 1}, # log_f

    ]

# Default centre of the initial ball of walkers, in the order I0, Re,
# n, sky, log_f.
default_starting_vector = np.array([10, 10, 1, 2.0, -2.0])

class PriorSetter:
    """
    Log-prior built from a list of per-parameter prior dictionaries.

    Only uniform priors are implemented. Each dictionary has the keys
    `type` (the string `'uniform'`), `min` and `max`, and the list is
    ordered like the parameter vector of the model being fitted.

    Attributes
    ----------
    prior_dict_list : list of dict
        The prior dictionaries, one per parameter.
    """

    def __init__(self, prior_dict_list):
        """
        Store the list of prior dictionaries.

        Parameters
        ----------
        prior_dict_list : list of dict
            One dictionary per parameter, in the order of the
            parameter vector, each with the keys `type`, `min` and
            `max`.
        """
        self.prior_dict_list = prior_dict_list

    def log_prior(self, theta):
        """
        Return the log-prior of a parameter vector.

        Parameters and priors are paired with `zip`, so if the two
        sequences have different lengths the unpaired elements are
        ignored. A uniform prior contributes 0 when the parameter is
        strictly inside `(min, max)` and `-inf` otherwise. Priors of
        any other `type` contribute nothing.

        Parameters
        ----------
        theta : array_like
            The parameter vector.

        Returns
        -------
        float
            0 if every parameter is inside its limits, `-inf`
            otherwise.
        """

        prior = 0

        for parameter, parameter_prior in zip(theta, self.prior_dict_list):

            if parameter_prior['type'] == 'uniform':
                if parameter_prior['min'] < parameter < parameter_prior['max']:
                    continue

                else:
                    prior += -np.inf

        return prior


class LogProbability_SersicSkyIS:
    """
    Log-probability of a Sersic + sky + intrinsic-scatter 1D model.

    The parameter vector is `theta = (I0, Re, n, sky, log_f)`. The
    model profile is the sum of a `Sersic` profile and a constant
    `Sky` level, and the likelihood is Gaussian with a variance given
    by the measurement errors plus an intrinsic scatter proportional
    to the model (`IntrinsicScatter`). The object is callable through
    `evaluate`, which is the function passed to `emcee`.

    Attributes
    ----------
    radius : numpy.ndarray
        The radii of the profile.
    data : numpy.ndarray
        The measured intensities.
    data_err : numpy.ndarray
        The uncertainties on the intensities.
    mask : numpy.ndarray
        Boolean array, `True` for the points excluded from the fit.
    prior : PriorSetter
        The prior of the model parameters.
    kernel : None
        The 1D convolution kernel. It is always set to `None`, so the
        model is never convolved.
    model_data_list : list of numpy.ndarray
        The model profile of every likelihood evaluation, appended in
        call order.
    model_data : numpy.ndarray
        The model profile of the last likelihood evaluation, on the
        unmasked points.
    residual_map : numpy.ndarray
        The data minus the model of the last likelihood evaluation, on
        the unmasked points.
    """

    def __init__(self, radius, data, data_err, mask, prior:PriorSetter, kernel=None):
        """
        Store the profile to fit and the prior.

        Parameters
        ----------
        radius : numpy.ndarray
            The radii of the profile, shape `(n_points,)`, in the
            units in which `Re` is to be measured.
        data : numpy.ndarray
            The measured intensities, shape `(n_points,)`.
        data_err : numpy.ndarray
            The 1 sigma uncertainties on the intensities, shape
            `(n_points,)`, in the units of `data`.
        mask : numpy.ndarray
            Boolean array of shape `(n_points,)`, `True` for the
            points excluded from the fit.
        prior
            The prior on `(I0, Re, n, sky, log_f)`.
        kernel : numpy.ndarray, optional
            A 1D convolution kernel. The argument is accepted but not
            stored: the `kernel` attribute is set to `None`.
        """

        self.radius = radius
        self.data = data
        self.data_err = data_err
        self.mask = mask
        self.prior = prior

        # self.radius_from_center = np.insert(radius, 0, 0.0)

        self.kernel = None

        self.model_data_list = []

    def log_likelihood(self, theta):
        """
        Return the Gaussian log-likelihood of a parameter vector.

        The model `m` is the Sersic profile plus the sky level,
        evaluated on the unmasked radii (if a kernel were set, the
        model would be evaluated on all the radii, convolved with
        `astropy.convolution.convolve` and then masked). The variance
        of each point is

            s^2 = data_err^2 + m^2 * exp(2 * log_f)

        and the log-likelihood is

            -0.5 * sum((data - m)^2 / s^2 + log(2 * pi * s^2)).

        The model, the residuals and the history of models are stored
        in `model_data`, `residual_map` and `model_data_list`.

        Parameters
        ----------
        theta : array_like
            The parameter vector `(I0, Re, n, sky, log_f)`.

        Returns
        -------
        float
            The log-likelihood.
        """

        sersic_instance = Sersic(
            I0 = theta[0],
            Re = theta[1],
            n = theta[2],
        )

        sky_instance = Sky(
            sky = theta[3]
        )

        is_instance = IntrinsicScatter(
            log_f = theta[4]
        )

        if self.kernel is not None:
            model_data_from_center = sersic_instance.model_from_radius(self.radius) + sky_instance.model_from_radius(self.radius)

            model_data_from_center = convolve(model_data_from_center, self.kernel)

            self.model_data = model_data_from_center[~self.mask]

        else:
            self.model_data = sersic_instance.model_from_radius(self.radius[~self.mask]) + sky_instance.model_from_radius(self.radius[~self.mask])

        self.model_data_list.append(self.model_data)

        # Measurement errors and intrinsic scatter added in quadrature.
        total_scatter = np.sqrt(self.data_err[~self.mask]**2 + self.model_data**2 * np.exp(2 * is_instance.log_f))

        self.residual_map = self.data[~self.mask] - self.model_data
        normalized_residual_map = self.residual_map / total_scatter

        chi_squared_map = normalized_residual_map**2
        chi_squared = np.sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * total_scatter**2.0))

        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood

    @property
    def info(self):
        """Print the order of the model parameters and the prior."""
        print('the parameters are: I0, Re, n, sky, log_f in the specific order. The prior is uniform for all parameters')

    def evaluate(self, theta):
        """
        Return the log-posterior of a parameter vector.

        The likelihood is evaluated only when the log-prior is finite.

        Parameters
        ----------
        theta : array_like
            The parameter vector `(I0, Re, n, sky, log_f)`.

        Returns
        -------
        float
            The log-prior plus the log-likelihood, or `-inf` if the
            parameters are outside the prior limits.
        """

        lp = self.prior.log_prior(theta)

        if not np.isfinite(lp):
            return -np.inf

        else:
            return lp + self.log_likelihood(theta)



class FitProfile(LogProbability_SersicSkyIS):
    """
    MCMC fit of a 1D profile with a Sersic + sky + scatter model.

    The class samples the log-posterior of
    `LogProbability_SersicSkyIS` with an `emcee.EnsembleSampler` and
    offers helpers to inspect and store the chains. The 5 parameters
    are `I0`, `Re`, `n`, `sky` and `log_f`. `FitProfile_noSky` is the
    4-parameter variant without the sky level.

    The typical sequence is `run`, `get_flat_chains`, then
    `plot_corner`, `ml_sample` or `save_samples`.

    Attributes
    ----------
    prior : PriorSetter or None
        The prior used by `evaluate` (see `__init__`).
    starting_vector : numpy.ndarray
        The centre of the initial ball of walkers.
    sampler : emcee.EnsembleSampler
        The sampler, available after `run`.
    flat_samples : numpy.ndarray
        The flattened chains, shape `(n_samples, 5)`, available after
        `get_flat_chains`.
    """

    def __init__(self,
                 radius,
                 data,
                 data_err,
                 mask,
                 prior=None,
                 starting_vector=None,
                 kernel=None):
        """
        Store the profile, the prior and the starting point.

        The prior attribute is first set to the default (or to the
        input one), and then assigned again by the parent `__init__`,
        which receives the `prior` argument as given: when `prior` is
        `None` the attribute therefore ends up being `None`.

        Parameters
        ----------
        radius : numpy.ndarray
            The radii of the profile, shape `(n_points,)`.
        data : numpy.ndarray
            The measured intensities, shape `(n_points,)`.
        data_err : numpy.ndarray
            The 1 sigma uncertainties on the intensities, shape
            `(n_points,)`.
        mask : numpy.ndarray
            Boolean array of shape `(n_points,)`, `True` for the
            points excluded from the fit.
        prior : PriorSetter, optional
            The prior on `(I0, Re, n, sky, log_f)`. `None` requests a
            `PriorSetter` built from `default_prior_dict_list`.
        starting_vector : numpy.ndarray, optional
            The parameter vector around which the walkers are
            initialised. `None` means `default_starting_vector`.
        kernel : numpy.ndarray, optional
            A 1D convolution kernel, passed to the parent class, which
            does not store it.
        """

        if prior is None:
            self.prior = PriorSetter(default_prior_dict_list)
        else:
            self.prior = prior

        if starting_vector is None:
            self.starting_vector = default_starting_vector
        else:
            self.starting_vector = starting_vector

        super().__init__(radius, data, data_err, mask, prior, kernel)


    def run(self, nsteps=1000, progress=True):
        """
        Run the MCMC sampling with `emcee`.

        32 walkers are initialised in a Gaussian ball of standard
        deviation 1e-5 around `starting_vector`, a new
        `emcee.EnsembleSampler` is created on `evaluate` and run for
        `nsteps` steps. The sampler is stored in `sampler`.

        Parameters
        ----------
        nsteps : int
            The number of MCMC steps of each walker.
        progress : bool
            If `True` a progress bar is shown.
        """

        pos = self.starting_vector + 1e-5 * np.random.randn(32, len(self.starting_vector))
        nwalkers, ndim = pos.shape

        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.evaluate)

        if not hasattr(self, 'sampler'):
            self.initialize()

        self.sampler.run_mcmc(pos, nsteps, progress=progress)

    def plot_chains(self, fig, axs):
        """
        Plot the trace of every parameter along the chain.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure containing the axes (not used).
        axs : sequence of matplotlib.axes.Axes
            One axis per parameter, in the order `I0`, `Re`, `n`,
            `sky`, `log_f`.
        """

        samples = self.sampler.get_chain()

        labels = ["I0", "Re", "n", "sky", 'log_f']
        for i in range(self.sampler.ndim):
            ax = axs[i]
            ax.plot(samples[:, :, i], "C0", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axs[-1].set_xlabel("step number");

    def get_flat_chains(self, discard, thin):
        """
        Flatten the chains into the `flat_samples` attribute.

        Parameters
        ----------
        discard : int
            The number of initial (burn-in) steps to discard.
        thin : int
            Keep only one step every `thin`.
        """
        self.flat_samples = self.sampler.get_chain(discard=discard, thin=thin, flat=True)


    def plot_corner(self, fig, **kwargs):
        """
        Draw the corner plot of the flattened chains.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure the corner plot is drawn on.
        **kwargs
            Keyword arguments passed to `corner.corner`.

        Raises
        ------
        ValueError
            If `get_flat_chains` has not been called.
        """
        if not hasattr(self, 'flat_samples'):
            raise ValueError("You need to call get_flat_chains() before plotting the corner plot.")

        fig = corner.corner(self.flat_samples, fig=fig, labels=["I0", "Re", "n", "sky", 'log_f'], **kwargs)

    @property
    def ml_sample(self):
        """
        Return the median of the flattened chains for every parameter.

        Despite the name, the value returned is the marginal median
        (the 0.5 quantile, ignoring NaNs) of each parameter and not the
        sample with the maximum likelihood.

        Returns
        -------
        numpy.ndarray
            The medians, in the order `I0`, `Re`, `n`, `sky`, `log_f`.

        Raises
        ------
        ValueError
            If `get_flat_chains` has not been called.
        """
        if not hasattr(self, 'flat_samples'):
            raise ValueError("You need to call get_flat_chains() before getting the ML fit.")

        return np.nanquantile(self.flat_samples, 0.5, axis=0)

    def convert_samples(self, func):
        """
        Replace the flattened chains with a transformed version.

        Parameters
        ----------
        func : callable
            A function that takes the `flat_samples` array and returns
            the converted array (e.g. a change of units).
        """
        self.flat_samples = func(self.flat_samples)

    def save_samples(self, filename):
        """
        Save the flattened chains to a `.npy` file.

        Parameters
        ----------
        filename : str
            The output file name, passed to `numpy.save`.

        Raises
        ------
        ValueError
            If `get_flat_chains` has not been called.
        """
        if not hasattr(self, 'flat_samples'):
            raise ValueError("You need to call get_flat_chains() before saving the samples.")

        np.save(filename, self.flat_samples)



class LogProbability_SersicIS:
    """
    Log-probability of a Sersic + intrinsic-scatter 1D model.

    This is the counterpart of `LogProbability_SersicSkyIS` without
    the sky level: the parameter vector is `theta = (I0, Re, n,
    log_f)` and the model profile is the `Sersic` profile alone. The
    likelihood is Gaussian with a variance given by the measurement
    errors plus an intrinsic scatter proportional to the model.

    Attributes
    ----------
    radius : numpy.ndarray
        The radii of the profile.
    data : numpy.ndarray
        The measured intensities.
    data_err : numpy.ndarray
        The uncertainties on the intensities.
    mask : numpy.ndarray
        Boolean array, `True` for the points excluded from the fit.
    prior : PriorSetter
        The prior of the model parameters.
    kernel : None
        The 1D convolution kernel. It is always set to `None`, so the
        model is never convolved.
    model_data_list : list of numpy.ndarray
        The model profile of every likelihood evaluation, appended in
        call order.
    model_data : numpy.ndarray
        The model profile of the last likelihood evaluation, on the
        unmasked points.
    residual_map : numpy.ndarray
        The data minus the model of the last likelihood evaluation, on
        the unmasked points.
    """

    def __init__(self, radius, data, data_err, mask, prior:PriorSetter, kernel=None):
        """
        Store the profile to fit and the prior.

        Parameters
        ----------
        radius : numpy.ndarray
            The radii of the profile, shape `(n_points,)`, in the
            units in which `Re` is to be measured.
        data : numpy.ndarray
            The measured intensities, shape `(n_points,)`.
        data_err : numpy.ndarray
            The 1 sigma uncertainties on the intensities, shape
            `(n_points,)`, in the units of `data`.
        mask : numpy.ndarray
            Boolean array of shape `(n_points,)`, `True` for the
            points excluded from the fit.
        prior
            The prior on `(I0, Re, n, log_f)`.
        kernel : numpy.ndarray, optional
            A 1D convolution kernel. The argument is accepted but not
            stored: the `kernel` attribute is set to `None`.
        """

        self.radius = radius
        self.data = data
        self.data_err = data_err
        self.mask = mask
        self.prior = prior

        self.kernel = None

        self.model_data_list = []

    def log_likelihood(self, theta):
        """
        Return the Gaussian log-likelihood of a parameter vector.

        The model `m` is the Sersic profile evaluated on the unmasked
        radii (if a kernel were set, the model would be evaluated on
        all the radii, convolved with `astropy.convolution.convolve`
        and then masked). The variance of each point is

            s^2 = data_err^2 + m^2 * exp(2 * log_f)

        and the log-likelihood is

            -0.5 * sum((data - m)^2 / s^2 + log(2 * pi * s^2)).

        The model, the residuals and the history of models are stored
        in `model_data`, `residual_map` and `model_data_list`.

        Parameters
        ----------
        theta : array_like
            The parameter vector `(I0, Re, n, log_f)`.

        Returns
        -------
        float
            The log-likelihood.
        """

        sersic_instance = Sersic(
            I0 = theta[0],
            Re = theta[1],
            n = theta[2],
        )

        is_instance = IntrinsicScatter(
            log_f = theta[3]
        )

        if self.kernel is not None:
            model_data_from_center = sersic_instance.model_from_radius(self.radius)

            model_data_from_center = convolve(model_data_from_center, self.kernel)

            self.model_data = model_data_from_center[~self.mask]

        else:
            self.model_data = sersic_instance.model_from_radius(self.radius[~self.mask])


        self.model_data_list.append(self.model_data)

        # Measurement errors and intrinsic scatter added in quadrature.
        total_scatter = np.sqrt(self.data_err[~self.mask]**2 + self.model_data**2 * np.exp(2 * is_instance.log_f))

        self.residual_map = self.data[~self.mask] - self.model_data
        normalized_residual_map = self.residual_map / total_scatter

        chi_squared_map = normalized_residual_map**2
        chi_squared = np.sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * total_scatter**2.0))

        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood

    @property
    def info(self):
        """Print the order of the model parameters and the prior."""
        print('the parameters are: I0, Re, n, log_f in the specific order. The prior is uniform for all parameters')

    def evaluate(self, theta):
        """
        Return the log-posterior of a parameter vector.

        The likelihood is evaluated only when the log-prior is finite.

        Parameters
        ----------
        theta : array_like
            The parameter vector `(I0, Re, n, log_f)`.

        Returns
        -------
        float
            The log-prior plus the log-likelihood, or `-inf` if the
            parameters are outside the prior limits.
        """

        lp = self.prior.log_prior(theta)

        if not np.isfinite(lp):
            return -np.inf

        else:
            return lp + self.log_likelihood(theta)


class FitProfile_noSky(LogProbability_SersicIS):
    """
    MCMC fit of a 1D profile with a Sersic + scatter model (no sky).

    The class is the counterpart of `FitProfile` for a profile from
    which the sky has already been removed: it samples the
    log-posterior of `LogProbability_SersicIS`, whose 4 parameters are
    `I0`, `Re`, `n` and `log_f`, instead of the 5 parameters `I0`,
    `Re`, `n`, `sky`, `log_f` of `FitProfile`. The methods are
    otherwise the same.

    The defaults of `__init__` are shared with `FitProfile`:
    `default_prior_dict_list` and `default_starting_vector` both have
    5 elements, ordered as `I0`, `Re`, `n`, `sky`, `log_f`. A prior
    and a starting vector with 4 elements, ordered as `I0`, `Re`, `n`,
    `log_f`, are expected by the likelihood of this class.

    Attributes
    ----------
    prior : PriorSetter or None
        The prior used by `evaluate` (see `__init__`).
    starting_vector : numpy.ndarray
        The centre of the initial ball of walkers.
    sampler : emcee.EnsembleSampler
        The sampler, available after `run`.
    flat_samples : numpy.ndarray
        The flattened chains, shape `(n_samples, n_parameters)`,
        available after `get_flat_chains`.
    """
    def __init__(self,
                 radius,
                 data,
                 data_err,
                 mask,
                 prior=None,
                 starting_vector=None,
                 kernel=None):
        """
        Store the profile, the prior and the starting point.

        The prior attribute is first set to the default (or to the
        input one), and then assigned again by the parent `__init__`,
        which receives the `prior` argument as given: when `prior` is
        `None` the attribute therefore ends up being `None`.

        Parameters
        ----------
        radius : numpy.ndarray
            The radii of the profile, shape `(n_points,)`.
        data : numpy.ndarray
            The measured intensities, shape `(n_points,)`.
        data_err : numpy.ndarray
            The 1 sigma uncertainties on the intensities, shape
            `(n_points,)`.
        mask : numpy.ndarray
            Boolean array of shape `(n_points,)`, `True` for the
            points excluded from the fit.
        prior : PriorSetter, optional
            The prior on `(I0, Re, n, log_f)`. `None` requests a
            `PriorSetter` built from the 5-element
            `default_prior_dict_list`.
        starting_vector : numpy.ndarray, optional
            The parameter vector around which the walkers are
            initialised; its length sets the number of sampled
            dimensions. `None` means the 5-element
            `default_starting_vector`.
        kernel : numpy.ndarray, optional
            A 1D convolution kernel, passed to the parent class, which
            does not store it.
        """

        if prior is None:
            self.prior = PriorSetter(default_prior_dict_list)
        else:
            self.prior = prior

        if starting_vector is None:
            self.starting_vector = default_starting_vector
        else:
            self.starting_vector = starting_vector

        super().__init__(radius, data, data_err, mask, prior, kernel)


    def run(self, nsteps=1000, progress=True):
        """
        Run the MCMC sampling with `emcee`.

        32 walkers are initialised in a Gaussian ball of standard
        deviation 1e-5 around `starting_vector`, a new
        `emcee.EnsembleSampler` is created on `evaluate` and run for
        `nsteps` steps. The sampler is stored in `sampler`.

        Parameters
        ----------
        nsteps : int
            The number of MCMC steps of each walker.
        progress : bool
            If `True` a progress bar is shown.
        """

        pos = self.starting_vector + 1e-5 * np.random.randn(32, len(self.starting_vector))
        nwalkers, ndim = pos.shape

        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.evaluate)

        if not hasattr(self, 'sampler'):
            self.initialize()

        self.sampler.run_mcmc(pos, nsteps, progress=progress)

    def plot_chains(self, fig, axs):
        """
        Plot the trace of every parameter along the chain.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure containing the axes (not used).
        axs : sequence of matplotlib.axes.Axes
            One axis per parameter, in the order `I0`, `Re`, `n`,
            `log_f`.
        """

        samples = self.sampler.get_chain()

        labels = ["I0", "Re", "n", 'log_f']
        for i in range(self.sampler.ndim):
            ax = axs[i]
            ax.plot(samples[:, :, i], "C0", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axs[-1].set_xlabel("step number");

    def get_flat_chains(self, discard, thin):
        """
        Flatten the chains into the `flat_samples` attribute.

        Parameters
        ----------
        discard : int
            The number of initial (burn-in) steps to discard.
        thin : int
            Keep only one step every `thin`.
        """
        self.flat_samples = self.sampler.get_chain(discard=discard, thin=thin, flat=True)


    def plot_corner(self, fig, **kwargs):
        """
        Draw the corner plot of the flattened chains.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure the corner plot is drawn on.
        **kwargs
            Keyword arguments passed to `corner.corner`.

        Raises
        ------
        ValueError
            If `get_flat_chains` has not been called.
        """
        if not hasattr(self, 'flat_samples'):
            raise ValueError("You need to call get_flat_chains() before plotting the corner plot.")

        fig = corner.corner(self.flat_samples, fig=fig, labels=["I0", "Re", "n", 'log_f'], **kwargs)

    @property
    def ml_sample(self):
        """
        Return the median of the flattened chains for every parameter.

        Despite the name, the value returned is the marginal median
        (the 0.5 quantile, ignoring NaNs) of each parameter and not the
        sample with the maximum likelihood.

        Returns
        -------
        numpy.ndarray
            The medians, in the order of the parameter vector.

        Raises
        ------
        ValueError
            If `get_flat_chains` has not been called.
        """
        if not hasattr(self, 'flat_samples'):
            raise ValueError("You need to call get_flat_chains() before getting the ML fit.")

        return np.nanquantile(self.flat_samples, 0.5, axis=0)

    def convert_samples(self, func):
        """
        Replace the flattened chains with a transformed version.

        Parameters
        ----------
        func : callable
            A function that takes the `flat_samples` array and returns
            the converted array (e.g. a change of units).
        """
        self.flat_samples = func(self.flat_samples)

    def save_samples(self, filename):
        """
        Save the flattened chains to a `.npy` file.

        Parameters
        ----------
        filename : str
            The output file name, passed to `numpy.save`.

        Raises
        ------
        ValueError
            If `get_flat_chains` has not been called.
        """
        if not hasattr(self, 'flat_samples'):
            raise ValueError("You need to call get_flat_chains() before saving the samples.")

        np.save(filename, self.flat_samples)
