class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'tcs':
            #return '/mnt/hdd/albin_zeqiri/ma/dataset/logs/'
            #return 'C:/Users/mi/Documents/Dataset/out/'
            #return "D:/csv-merger/out/"
            #return "/mnt/hdd/sebastian/chi2023/dataset/"
            return "H:/data/ThermalDataset/"
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
