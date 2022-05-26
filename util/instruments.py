import numpy as np
import QuantLib as ql

def makeSwap(start, maturity, notional, fixedRate, index, tipo = ql.VanillaSwap.Payer):
    """
    Returns the Quantlib swap object
    """
    fixedLegTenor = ql.Period("1y")
    fixedLegBDC = ql.ModifiedFollowing
    floatLegTenor = index.tenor()
    fixedLegDC = index.dayCounter() #ql.Thirty360(ql.Thirty360.BondBasis)
    spread = 0.0
    fixedSchedule = ql.Schedule(start, maturity, fixedLegTenor, index.fixingCalendar(), fixedLegBDC, fixedLegBDC, 
                                   ql.DateGeneration.Backward, False)
    floatSchedule = ql.Schedule(start, maturity, floatLegTenor, index.fixingCalendar(), index.businessDayConvention(),
                                    index.businessDayConvention(), ql.DateGeneration.Backward,False)
    swap = ql.VanillaSwap(tipo, notional,fixedSchedule,fixedRate,fixedLegDC,floatSchedule,index,spread,index.dayCounter())
    return swap

