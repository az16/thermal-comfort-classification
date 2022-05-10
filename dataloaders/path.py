class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'tcs':
            return '~/ma/dataset/logs/'
            #return "D:/csv-merger/out/"
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
