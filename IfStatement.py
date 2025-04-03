temparature = int (input("Input temperature:"))
if temparature > 30:
    print ("It's a hot day")
    print ("Drink 2 much water")
elif temparature > 20: # 20 < temperature < 30
    print ("nice weather")
elif temparature > 10: # 10 < temperature < 20
    print ("cold a bit")
else: print("2 cold")
print ("Done")