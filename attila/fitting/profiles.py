import numpy as np
import matplotlib.pyplot as plt

from astropy.convolution import convolve
import emcee
import corner

class Sersic:

    def __init__(self, I0, Re, n):
        
        self.I0 = I0
        self.Re = Re
        self.n = n

    @property
    def Sersic_constant(self):

        """
        Straight from autogalaxy
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
        return (self.I0 * np.exp(-self.Sersic_constant * (((radius / self.Re) ** (1.0 / self.n)))))
    
class Sky:

    def __init__(self, sky):
        self.sky = sky

    def model_from_radius(self, radius):
        return np.ones(radius.shape) * self.sky
    
class IntrinsicScatter:

    def __init__(self, log_f):
        self.log_f = log_f


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

default_starting_vector = np.array([10, 10, 1, 2.0, -2.0])

class PriorSetter:

    def __init__(self, prior_dict_list):
        self.prior_dict_list = prior_dict_list

    def log_prior(self, theta):

        prior = 0

        for parameter, parameter_prior in zip(theta, self.prior_dict_list):

            if parameter_prior['type'] == 'uniform':
                if parameter_prior['min'] < parameter < parameter_prior['max']:
                    continue

                else:
                    prior += -np.inf

        return prior


class LogProbability_SersicSkyIS:

    def __init__(self, radius, data, data_err, mask, prior:PriorSetter, kernel=None):

        self.radius = radius
        self.data = data
        self.data_err = data_err
        self.mask = mask
        self.prior = prior

        # self.radius_from_center = np.insert(radius, 0, 0.0) 

        self.kernel = None

        self.model_data_list = []

    def log_likelihood(self, theta):

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
        print('the parameters are: I0, Re, n, sky, log_f in the specific order. The prior is uniform for all parameters')

    def evaluate(self, theta):

        lp = self.prior.log_prior(theta)

        if not np.isfinite(lp):
            return -np.inf
        
        else:
            return lp + self.log_likelihood(theta)



class FitProfile(LogProbability_SersicSkyIS):

    def __init__(self, 
                 radius, 
                 data, 
                 data_err, 
                 mask,
                 prior=None, 
                 starting_vector=None, 
                 kernel=None):

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

        pos = self.starting_vector + 1e-5 * np.random.randn(32, len(self.starting_vector))
        nwalkers, ndim = pos.shape

        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.evaluate)

        if not hasattr(self, 'sampler'):
            self.initialize()

        self.sampler.run_mcmc(pos, nsteps, progress=progress)

    def plot_chains(self, fig, axs):

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
        self.flat_samples = self.sampler.get_chain(discard=discard, thin=thin, flat=True)


    def plot_corner(self, fig, **kwargs):
        if not hasattr(self, 'flat_samples'):
            raise ValueError("You need to call get_flat_chains() before plotting the corner plot.")

        fig = corner.corner(self.flat_samples, fig=fig, labels=["I0", "Re", "n", "sky", 'log_f'], **kwargs)

    @property
    def ml_sample(self):
        if not hasattr(self, 'flat_samples'):
            raise ValueError("You need to call get_flat_chains() before getting the ML fit.")

        return np.nanquantile(self.flat_samples, 0.5, axis=0)

    def convert_samples(self, func):
        self.flat_samples = func(self.flat_samples)

    def save_samples(self, filename):
        if not hasattr(self, 'flat_samples'):
            raise ValueError("You need to call get_flat_chains() before saving the samples.")

        np.save(filename, self.flat_samples)



class LogProbability_SersicIS:

    def __init__(self, radius, data, data_err, mask, prior:PriorSetter, kernel=None):

        self.radius = radius
        self.data = data
        self.data_err = data_err
        self.mask = mask
        self.prior = prior

        self.kernel = None

        self.model_data_list = []

    def log_likelihood(self, theta):

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
        print('the parameters are: I0, Re, n, log_f in the specific order. The prior is uniform for all parameters')

    def evaluate(self, theta):

        lp = self.prior.log_prior(theta)

        if not np.isfinite(lp):
            return -np.inf
        
        else:
            return lp + self.log_likelihood(theta)


class FitProfile_noSky(LogProbability_SersicIS):
    def __init__(self, 
                 radius, 
                 data, 
                 data_err, 
                 mask,
                 prior=None, 
                 starting_vector=None, 
                 kernel=None):

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

        pos = self.starting_vector + 1e-5 * np.random.randn(32, len(self.starting_vector))
        nwalkers, ndim = pos.shape

        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.evaluate)

        if not hasattr(self, 'sampler'):
            self.initialize()

        self.sampler.run_mcmc(pos, nsteps, progress=progress)

    def plot_chains(self, fig, axs):

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
        self.flat_samples = self.sampler.get_chain(discard=discard, thin=thin, flat=True)


    def plot_corner(self, fig, **kwargs):
        if not hasattr(self, 'flat_samples'):
            raise ValueError("You need to call get_flat_chains() before plotting the corner plot.")

        fig = corner.corner(self.flat_samples, fig=fig, labels=["I0", "Re", "n", 'log_f'], **kwargs)

    @property
    def ml_sample(self):
        if not hasattr(self, 'flat_samples'):
            raise ValueError("You need to call get_flat_chains() before getting the ML fit.")

        return np.nanquantile(self.flat_samples, 0.5, axis=0)

    def convert_samples(self, func):
        self.flat_samples = func(self.flat_samples)

    def save_samples(self, filename):
        if not hasattr(self, 'flat_samples'):
            raise ValueError("You need to call get_flat_chains() before saving the samples.")

        np.save(filename, self.flat_samples)