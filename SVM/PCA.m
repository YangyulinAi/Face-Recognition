function [pcaA ,V]=PCA(A,k)
[m,n]=size(A);

meanVec=mean(A);  % Sample Means
 
Z= ( A-repmat(meanVec,m,1)  );  
                                
covMatT =Z*Z'; % PCA Extraction

% Calculate the first k eigenvalues
[V, ~]=eigs(covMatT,k);  %V is m*k
 
 
V=Z'*V;  % Get the feature vector
 
for i=1:k
    V(:,i)=V(:,i)/norm(V(:,i));  %norm is the parametric number, default is 2
end
 

pcaA=Z*V; % Dimensionality reduction
end
