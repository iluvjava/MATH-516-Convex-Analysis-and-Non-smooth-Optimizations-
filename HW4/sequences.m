close all;
a = 1;
Points = 0:300;
for Index = Points(2:end)
    p = a(end); % Previous 
    a(end + 1) = (sqrt(p^4 + 4*p^2) - p^2)/2;
end
figure; 
plot(a, '-o', "linewidth", 1); hold on
UpperBound = 2./(Points + 2);
LowerBound = 2./(Points + 3);
plot(UpperBound, "linewidth", 1);
plot(LowerBound, "linewidth", 1);
legend(["sequence", "upper bound", "Lower Bound"])

figure; 
DiffBand = UpperBound - LowerBound; 
loglog(DiffBand);
Coeff = polyfit(log(Points(2:end)), log(DiffBand(2:end)), 1);
disp("Log log Coefficient:")
disp(num2str(Coeff(1)))

%% 
figure; 
WhatBound = (sqrt(LowerBound.^4 + 4*LowerBound.^2) - LowerBound.^2)/2;
plot(a(2:end)); hold on 
plot(WhatBound)

