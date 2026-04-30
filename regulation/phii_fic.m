function result=phii_fic(x,m)
g=0.087;
I=177.;
c=615.; 
y=c*x-I;  % y=c*( x+(m-1) )-I;
 if y~=0
  result = m.*y./(1-exp(-g*m.*y));
 else
  result=0;
 end
end