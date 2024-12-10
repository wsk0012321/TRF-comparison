function [strainSet, rtrainSet, stestSet, rtestSet] = trf_partition(stim,resp,nfold,testTrial)
      strainSet = [], rtrainSet = []; stestSet = []; rtestSet = [];
      for i = 1:length(resp)
          % extract data of each subject and transpose it
          r = cell2mat(resp(i))';
          s = cell2mat(stim(i))';
          [strain,rtrain,stest,rtest] = mTRFpartition(s,r,nfold,testTrial);
          % mTRFpartition returns different types for train and test data:
          % cell for the train data and numeric data for the testset
          % rather than append the sets in batches, we append subsets one
          % by one
          for n = 1:length(strain)
              strainSet = [strainSet, strain(n)];
              rtrainSet = [rtrainSet, rtrain(n)];
          end
          stestSet = [stestSet,{stest}];
          rtestSet = [rtestSet,{rtest}];
      end
end