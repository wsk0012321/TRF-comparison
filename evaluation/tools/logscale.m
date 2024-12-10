function data = logscale(data)
      for i = 1:length(data)
          val = data(i);
          if val ~= 0
              data(i) = log(val);
          end
      end     
end