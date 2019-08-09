phy_score = [15 , 12 , 8  , 8 ,  7 ,  7 ,  7 ,  6   , 5  , 3]
his_score = [10  ,25 , 17  ,11 , 13 , 17 , 20 , 13 , 9  , 15]
n=10
phy_total =0
his_total =0
mul = 0
phy1=0
for i in range (0,n):
    phy_total+=phy_score[i]
    his_total+=his_score[i]
    mul += phy_score[i]*his_score[i]
    phy1 += phy_score[i]*phy_score[i]
ans = (n*mul - phy_total*his_total)/(n*phy1 - phy_total*phy_total)
print('%.3f'%ans)
