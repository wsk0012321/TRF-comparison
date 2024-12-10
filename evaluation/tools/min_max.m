function normalizedData = min_max(data)
   minValue = min(data);
   maxValue = max(data);
   normalizedData = (data - minValue) / (maxValue - minValue);
end