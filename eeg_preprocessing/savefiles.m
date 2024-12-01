function savefiles(ALLEEG,ids,id_run)
    gendir = 'E:/PhD/data/Di_Liberto/transformed/Natural Speech/EEG/';
    savedir = strcat(gendir,'Run',num2str(id_run),'/interpolated/');
    for i = 1:length(ids)
        id = ids(i);
        dataset = ALLEEG(id);
        setname = ALLEEG(id).setname;
        matches = regexpi(setname, 'Subject\d+','match');
        subject = char(matches);
        filename = strcat(subject,'.set');
        pop_saveset(dataset, 'filename',filename,'filepath',savedir);
        fprintf('Save %s\n',filename);
    end
end