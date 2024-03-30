function [d] = Hellinger_dist(P, Q)
%This function computes the Hellinger distance between two distirbution matrices P and Q

sum = 0;
[m, n] = size(P);
for i=1:m
    for j=1:n
        p = P(i, j);
        q = Q(i, j);
        h2 = (sqrt(p) - sqrt(q))^2 +(sqrt(1-p) - sqrt(1-q))^2;
        sum = sum + h2; 
    end
end

d = sum/(m*n);

end
