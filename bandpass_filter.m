function ALLEEG = bandpass_filter(ALLEEG,ids)
    for i = 1:length(ids)
        id = ids(i);
        dataset = ALLEEG(id);
        setname = char(regexpi(dataset.setname,'Subject\d+','match'));
        dataset = pop_eegfiltnew(dataset, 1, 8);
        ALLEEG(id) = dataset;
        fprintf('Filtered dataset %s\n',setname);
    end
end