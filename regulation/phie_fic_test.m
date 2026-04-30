function result=phie_fic_test(x,m)
g=0.16;
I=125.;
c=310.;
y=c*( x+(m-1) )-I;
 if y~=0
  result = y./(1-exp(-g* y));
 else
  result=0;
 end
end