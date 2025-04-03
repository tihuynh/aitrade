weight = float (input ("Input your weight: "))
unit = input ("Your unit is Kg or Pound :")
if unit.upper() == "KG":
    conversion = weight / 0.45
    print("Your weight in Pound = "+str(conversion))
elif unit.upper() == "POUND":
    conversion = weight * 0.45
    print("Your weight in Kg = "+str(conversion))
print ("Done")