import numpy as np

def CVA_calculation(dPD, exposure, time_grid, t0_curve, nPath, recovery = 0.4):
    discounted_exposure = np.zeros(exposure.shape)
    discount_factors = np.zeros(len(time_grid))
    for i in range(len(time_grid)):
        discount_factors[i] = t0_curve.discount(i)
        discounted_exposure[:,i] = exposure[:,i]*discount_factors[i]
    positive_exposure = exposure.copy()
    negative_exposure = exposure.copy()
    positive_exposure_disc = discounted_exposure.copy()
    negative_exposure_disc = discounted_exposure.copy()
    positive_exposure[positive_exposure<0] = 0
    negative_exposure[negative_exposure>0] = 0
    positive_exposure_disc[positive_exposure_disc<0] = 0
    negative_exposure_disc[negative_exposure_disc>0] = 0
    # Expected positive exposure
    exp_p_exposure = np.sum(positive_exposure, axis = 0)/nPath
    exp_p_disc_exposure = np.sum(positive_exposure_disc, axis = 0)/nPath
    # Expected negative exposure
    exp_n_exposure = np.sum(negative_exposure, axis = 0)/nPath
    exp_n_disc_exposure = np.sum(negative_exposure_disc, axis = 0)/nPath
    CVA = (1-recovery) * np.sum(exp_p_disc_exposure[1:] * dPD)  # CVA del portafoglio
    return CVA, positive_exposure_disc, negative_exposure_disc