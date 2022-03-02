class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'tcs':
            return '/mnt/hdd/shared_datasets/tcs'
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
