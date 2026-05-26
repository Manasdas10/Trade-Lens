import sys

close_price = float(sys.argv[1])

if close_price > 25000:
    print("BUY")
elif close_price < 24000:
    print("SELL")
else:
    print("HOLD")