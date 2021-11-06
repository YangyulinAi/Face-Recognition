function svm = svmTrain(X,Y,kertype,C)
options = optimset;    
options.LargeScale = 'off';
options.Display = 'off';% there do not ouput

n = length(Y);%Length of the array Y
H = (Y'*Y).*kernel(X,X,kertype);

f = -ones(n,1); % f is 1*n of -1
A = [];
b = [];
Aeq = Y; 
beq = 0;
lb = zeros(n,1); 
ub = C*ones(n,1);
a0 = zeros(n,1);  % a0 is the initial approximation
[a,~,~,~,~]  = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);

C=100;
local=a>0&a<C;
sv_label = find(local);  % if 0<a<a(max)Then consider x as the support vector 
svm.a = a(sv_label);
svm.Xsv = X(:,sv_label);
svm.Ysv = Y(sv_label);
svm.svnum = length(sv_label);
%svm.label = sv_label;
end