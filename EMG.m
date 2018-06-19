function [h,m,Q] = EMG(flag, imgPath, K)
    %%Load the image and convert it to rgb
    [img cmap] = imread(imgPath);
    img_rgb = ind2rgb(img,cmap);
    orgRow = size(img,1);
    orgCol = size(img,2);
    %Reshape the rgb image to have all pixels in rows and rgb values in columns
    img = reshape(img_rgb,[size(img_rgb,1)*size(img_rgb,2) 3]);
    row = size(img,1);
    col = size(img,2);
    
    %Run K-means algorithm for 3 iterations and initialize the parameters of
    %EM algorithm from it
    [idx,C] = kmeans(img,K,'maxIter',3,'EmptyAction', 'singleton');
    m = C;
    pi = zeros(K,1);
    sigma = zeros(col,col,K);
    p = zeros(row,K);
    for i=1:K
        clusterPoints = find(idx==i);
        pi(i) = length(clusterPoints)/row;
        sigma(:,:,i) = cov(img(clusterPoints,:));
        p(:,i) = mvnpdf(img,m(i,:),sigma(:,:,i));
    end
    
    I = eye(3);
    %initilize Expected complete log likelihoods Q1(E step) and Q2(M Step)
    Q1=0;
    Q2=0;
    %Initialize responsibility
    gammaVal = zeros(row,K);
    %assign the smallest values to zeros to avoid errors while calculating log
    p(p==0) = 10^-100;
    pi(pi==0) = 10^-100;
    
    %Number of iterations
    for iter=1:105
        %iter
        %EXPECTATION STEP
        denom = 0;
        %Calculate denominator of responsibility
        for i=1:K
            denom = denom + pi(i)*p(:,i);
        end
        %Calculate numerator of responsibility
        for i=1:K
            gammaVal(:,i) = (pi(i)*p(:,i))./denom;
        end
        %Add the Expected complete log likelihood value after E step
        Q1 = [Q1 calcQ(K,flag,pi,p,gammaVal,sigma)];
        
        %MAXIMIZATION STEP
        for i=1:K
            %calculate mu,pi and sigmas after expectation step
            Ni = sum(gammaVal(:,i));
            m(i,:) = gammaVal(:,i)'*img/Ni;
            pi(i) = Ni/row;
            sigma(:,:,i) = 0;
            for j=1:row
                sigma(:,:,i) = sigma(:,:,i)+gammaVal(j,i)*(img(j,:)-m(i,:))'*(img(j,:)-m(i,:));
            end
            %If flag is 1 then add a regularization term lambda
            if flag==1
                lambda = 0.002;
                sigma(:,:,i) = (sigma(:,:,i)+lambda*I)/Ni;
            else
                sigma(:,:,i) = (sigma(:,:,i))/Ni;
            end
        end
        for i=1:K
            %rounded the values to avoid error while calculating mvnpdf
            s=round(sigma(:,:,i),10);
            p(:,i) = mvnpdf(img,m(i,:),s);
        end
        %assign the smallest values to zeros to avoid errors
        p(p==0) = 10^-100;
        pi(pi==0) = 10^-100;
        %Add the Expected complete log likelihood value after E step
        Q2 = [Q2 calcQ(K,flag,pi,p,gammaVal,sigma)];
        
    end
    Q=[Q1(2:length(Q1));Q2(2:length(Q2))];
    
    clusterAssgn=img;
    for i=1:row
        [mx mxInd] = max(gammaVal(i,:));
        %Assign the points to the cluster centre with highest
        %responsibility
        clusterAssgn(i,:) = m(mxInd,:);
    end
    %Plot the compressed image
    opImg = reshape(clusterAssgn,[orgRow orgCol 3]);
    subplot(132),imshow(opImg),title(sprintf('Compressed Image for K=%d',K));
    %Plot the expected complete log likelihood function value
    subplot(133),plot(1:105,Q(1,:),1:105,Q(2,:)),title(sprintf('Expected Complete Log Likelihood for K=%d',K));
    legend('E-Step','M-Step');
    xlabel('Iterations');
    ylabel('Expected Complete log likelihood');
    %Return N*1 vector which is the expected omplete log likelihood value
    %after M Step
    Q=Q(2,:)';
    %Return responsibility in h
    h=gammaVal;
end

%Calulate Expected Complete log likelihood function value
function [LC] = calcQ(K,flag,pi,p,gammaVal,sigma)
    LC=0;
    regTerm=0;
    for j=1:K
        temp = pi(j)*p(:,j);
        temp(temp==0)=10^-200;
        LC = LC + gammaVal(:,j)'*log(temp);
        regTerm = regTerm + trace(sigma(:,:,j));
    end
    %Add regularization term if flag==1
    if flag==1
        LC = LC+regTerm;
    end
end