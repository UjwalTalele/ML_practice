phy_score = [15 , 12 , 8  , 8 ,  7 ,  7 ,  7 ,  6   , 5  , 3]
his_score = [10  ,25 , 17  ,11 , 13 , 17 , 20 , 13 , 9  , 15]
n=10;
phy_total =0;
his_total =0;
mul = 0;
sq=0
for i in range (0,n):
    phy_total+=phy_score[i];
    his_total+=his_score[i];
    mul += phy_score[i]*his_score[i];
    sq += phy_score[i]*phy_score[i];
x_mean = float((phy_total)/n);
y_mean = float((his_total)/n);
slope = (n*mul - phy_total*his_total)/(n*sq - phy_total*phy_total);
constant = y_mean - (x_mean*slope);
ans = slope*10+constant;
print('%.1f'%ans);
