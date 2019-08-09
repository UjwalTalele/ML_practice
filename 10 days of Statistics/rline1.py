Learn more or give us feedback
import math
phy_score = [15 , 12 , 8  , 8 ,  7 ,  7 ,  7 ,  6   , 5  , 3]
his_score = [10  ,25 , 17  ,11 , 13 , 17 , 20 , 13 , 9  , 15]
n=10
his_total =0
phy_total =0
for i in range (0,n):
    phy_total=phy_total+phy_score[i];
    his_total=his_total+his_score[i];
phy_avg = float((phy_total)/n);
his_avg = float((his_total)/n);
mul=0;
temp1=0;
temp2=0;
for i in range (0,n):
  temp2 = temp2 + (phy_score[i]-phy_avg)**2;
  temp1 = temp1 + (his_score[i]-his_avg)**2;
  mul = mul + (phy_score[i]-phy_avg)*(his_score[i]-his_avg);
final_ans = float(mul)/math.sqrt(temp2*temp1);
print (final_ans) 
