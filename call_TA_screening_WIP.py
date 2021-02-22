import ta_screen_0_12 as ta_long
import time

tic = time.perf_counter()


ta_long.TA_screening('MGM')


toc = time.perf_counter()

tic_toc = (toc - tic) / 60

print(f"completed Pred in {tic_toc:0.4f} min")