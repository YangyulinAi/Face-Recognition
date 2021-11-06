function Y = svmTest(svm,Xt,kertype)
temp = (svm.a'.*svm.Ysv)*kernel(svm.Xsv,svm.Xsv,kertype);
total_b = svm.Ysv-temp;
b = mean(total_b);
w = (svm.a'.*svm.Ysv)*kernel(svm.Xsv,Xt',kertype);
Y = sign(w+b);
end

