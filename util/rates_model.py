import numpy as np
import QuantLib as ql 

daycounter = ql.Actual365Fixed()
def timeFromReferenceFactory(daycounter, ref):
    def impl(dat):
        return daycounter.yearFraction(ref, dat)
    return np.vectorize(impl)

def alpha(t, mean_rev, sigma, r0): # ok check fatto
    """ 
    This function returns the alpha 
    time-dependent parameter.
    α(t) = f(0, t) + 0.5(σ(1-exp(-kt))/k)^2

    Parameters:
    t : reference time in years.

    Returns:
    α(t) : deterministic parameter to recover term-rates.
    """
    discount1 = np.exp(-r0*t)
    discount2 = np.exp(-r0*(t+(1/365)))
    f = np.log(discount1/discount2)/(1/365)
    a_t = (sigma*sigma)*((1-np.exp(-mean_rev*t))**2)
    a_t /= (2*mean_rev*mean_rev)
    return f + 0.5*a_t

def gauss_params(r0, mean_rev, sigma, s, t, r_s):
    """ 
    This function returns the conditional mean
    and conditional variance of the short rate process 
    given the known value
    at time s <= t.
    E{r(t)|r(s)} = (r(s) - α(s))*exp(-k(t-s)) + α(t)
    Var{r(t)|r(s)} = σ^2[1 - exp(-2k(t-s))]/(2k)

    Parameters:
    s : information stopping time in years.
    t : reference time in years.
    r_s : short rate known at time s.

    Returns:
    E{r(t)|r(s)} : conditional mean
    Var{r(t)|r(s)} : conditional variance
    """
    decay_factor = np.exp(-mean_rev*(t-s))
    a_t = alpha(t, mean_rev, sigma, r0)
    a_s = alpha(s, mean_rev, sigma, r0)
    E_rt = (r_s - a_s)*decay_factor + a_t

    Var_rt = (1 - decay_factor**2)*sigma**2
    Var_rt /= 2*mean_rev

    return E_rt, Var_rt

def short_rate_path(mean_rev, sigma, times, r0):
    """ 
    This function returns a path drawn from
    the distribution of the conditional short rate.

    Parameters:
    path_length : time span in years.
    path_steps : number of steps in the discretized path.

    Returns:
    rt_path : array containing the short rate points.
    rt_path_std : array containing the standardized short rate points.
    
    """
    rt_path = np.zeros_like(times)
    rt_path_std = np.zeros_like(times)
    #discount1 = np.exp(-r0*0)
    #discount2 = np.exp(-r0*(1/365))
    rt_path[0] =  r0 #np.log(discount1/discount2)/(1/365)
    for step, st in enumerate(zip(times[:-1], times[1:]), 1):
        s, t = st
        Et, Vt = gauss_params(r0, mean_rev, sigma, s, t, rt_path[step-1])
        E_std, V_std = gauss_params(r0, mean_rev, sigma, 0, t, rt_path[0])
        rt = np.random.normal(loc=Et, scale=np.sqrt(Vt))
        rt_path[step] = rt
        rt_path_std[step] = (rt - E_std)/np.sqrt(V_std)
    return rt_path, rt_path_std

def A_B(S, T, r0, mean_rev, sigma):
    """ 
    This function returns the time dependent parameters
    of the ZCB, where S <= T.
    B(S, T) = (1 - exp(-k(T-S)))/k
    A(S, T) = P(0,T)/P(0,S) exp(B(S,T)f(0,S) - 
                σ^2(exp(-kT)-exp(-kS))^2(exp(2kS)-1)/(4k^3))

    Parameters :
    S : future reference time of the ZCB in years.
    T : future reference maturity of the ZCB years.

    Returns : 
    A(S, T) : scale factor of the ZCB
    B(S, T) : exponential factor of the ZCB
    """
    discount1 = np.exp(-r0*S)
    discount2 = np.exp(-r0*(S+(1/365)))
    
    f0S = np.log(discount1/discount2)/(1/365)
    P0T = np.exp(-r0*T)
    P0S = discount1

    B = 1 - np.exp(-mean_rev*(T - S))
    B /= mean_rev

    exponent = sigma*(np.exp(-mean_rev*T) - np.exp(-mean_rev*S))
    exponent *= exponent
    exponent *= np.exp(2*mean_rev*S) - 1
    exponent /= -4*(mean_rev**3)
    exponent += B*f0S
    A = np.exp(exponent)*P0T/P0S
    return A, B
    
def ZCB(S, T, rs, r0, mean_rev, sigma):
    """ 
    This function returns the price of a ZCB
    P(S, T) at future reference time S and maturity T 
    with S <= T.

    Parameters :
    S : future reference time of the ZCB in years.
    T : future reference maturity of the ZCB years. 

    Returns :
    P(S, T) : ZCB price with maturity T at future date S.
    """
    A, B = A_B(S, T, r0, mean_rev, sigma)
    return A*np.exp(-B*rs)



def getfixed(swap, data, today, date_grid):
    ###### fixed leg #####
    timeFromReference = timeFromReferenceFactory(daycounter, today)
    t = ql.Actual365Fixed().yearFraction(today,data)
    fixed_leg = swap.leg(0)
    n = len(fixed_leg)
    fixed_times=[]
    fixed_amounts=[]
    if n == len(date_grid):
        for i in range(n):
            cf = fixed_leg[i]
            cf_time = daycounter.yearFraction(date_grid[i], cf.date() )
            t_i = timeFromReference(cf.date())
            if t_i > t:
                fixed_times.append(t_i)
                fixed_amounts.append(cf.amount())
        fixed_times = np.array(fixed_times)
        fixed_amounts = np.array(fixed_amounts)
    else: 
        for i in range(n):
            cf = fixed_leg[i]
            t_i = timeFromReference(cf.date())
            if t_i > t:
                fixed_times.append(t_i)
                fixed_amounts.append(cf.amount())
        fixed_times = np.array(fixed_times)
        fixed_amounts = np.array(fixed_amounts)
    return fixed_times, fixed_amounts

def getfloating(swap, data, today):
    timeFromReference = timeFromReferenceFactory(daycounter, today)
    float_leg = swap.leg(1)
    t = ql.Actual365Fixed().yearFraction(today,data)
    n = len(float_leg)
    float_times = []
    float_dcf = []
    accrual_start_time = []
    accrual_end_time = []
    nominals = []
    for i in range(n):
        cf = ql.as_floating_rate_coupon(float_leg[i])
        value_date = cf.referencePeriodStart()
        t_fix_i = timeFromReference(value_date)
        t_i = timeFromReference(cf.date()) 
        if t_fix_i >= t:
            iborIndex = cf.index()
            index_mat = cf.referencePeriodEnd()
            float_dcf.append(cf.accrualPeriod())
            accrual_start_time.append(t_fix_i)
            accrual_end_time.append(timeFromReference(index_mat))
            float_times.append(t_i)
            nominals.append(cf.nominal())
    return np.array(float_times), np.array(float_dcf), np.array(accrual_start_time), np.array(accrual_end_time), np.array(nominals)

def swapPathNPV(swap, data, today, date_grid, mean_rev, sigma, r0):
    timeFromReference = timeFromReferenceFactory(daycounter, today)
    fixed_times, fixed_amounts = getfixed(swap, data, today, date_grid)
    float_times, float_dcf, accrual_start_time, accrual_end_time, nominals = getfloating(swap, data, today)
    t = ql.Actual365Fixed().yearFraction(today,data)
    df_times = np.concatenate([fixed_times, accrual_start_time, accrual_end_time, float_times])
    df_times = np.unique(df_times)
    fix_idx = np.in1d(df_times, fixed_times, True)
    fix_idx = fix_idx.nonzero()
    float_idx = np.in1d(df_times, float_times, True)
    float_idx = float_idx.nonzero()
    accrual_start_idx = np.in1d(df_times, accrual_start_time, True)
    accrual_start_idx = accrual_start_idx.nonzero()
    accrual_end_idx = np.in1d(df_times, accrual_end_time, True)
    accrual_end_idx = accrual_end_idx.nonzero()
    def calc(x_t):
        discount = np.vectorize(lambda T: ZCB(t,T, x_t, r0, mean_rev, sigma))
        dfs = discount(df_times)
        fix_leg_npv = np.sum(fixed_amounts * dfs[fix_idx])
        index_fixings = (dfs[accrual_start_idx] / dfs[accrual_end_idx] - 1) 
        index_fixings /= float_dcf
        float_leg_npv = np.sum(nominals * index_fixings * float_dcf * dfs[float_idx])
        npv = float_leg_npv - fix_leg_npv
        return npv ,float_leg_npv, fix_leg_npv
    return calc