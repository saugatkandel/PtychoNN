import numpy as np


#plt.matshow(np.log10(data_diffr[0,0]))
def read_experimental_data(data_path: str):
    try:
        data_diffr_red = np.load(data_path + '/20191008_39_diff_reduced.npz')['arr_0']
    except:
        data_diffr = np.load(data_path + '/20191008_39_diff.npz')['arr_0']
        data_diffr_red = np.zeros((data_diffr.shape[0],data_diffr.shape[1],64,64), float)
        for i in tqdm(range(data_diffr.shape[0])):
            for j in range(data_diffr.shape[1]):
                data_diffr_red[i,j] = resize(data_diffr[i,j,32:-32,32:-32],(64,64),preserve_range=True, anti_aliasing=True)
                data_diffr_red[i,j] = np.where(data_diffr_red[i,j]<3,0,data_diffr_red[i,j])
    
    real_space = np.load(data_path + '/20191008_39_amp_pha_10nm_full.npy')
    return data_diffr_red, real_space

def get_train_test_data(x_data: np.ndarray, y_data: np.ndarray, 
                        n_train_lines: int, n_test_lines: int,
                        img_h: int, img_w: int,
                        shuffle: bool = True,
                        random_state: int = 0,
                        out_dtype: str ='complex64'):
    
    from sklearn.utils import shuffle
    
    tst_strt = x_data.shape[0] - n_test_lines #Where to index from
    print('Indexing the test set from', tst_strt)
    
    if 'complex' in out_dtype:
        x_data = x_data + 0j
    
    X_train = x_data[:n_train_lines,:].reshape(-1, img_h, img_w)[...,np.newaxis]
    X_test = x_data[tst_strt:,tst_strt:].reshape(-1,img_h, img_w)[...,np.newaxis]
    Y_train = y_data[:n_train_lines, :].reshape(-1, img_h, img_w)[..., np.newaxis]
    Y_test = y_data[tst_strt:, tst_strt:].reshape(-1, img_h, img_w)[...,np.newaxis]
    
    ntrain = X_train.shape[0] * X_train.shape[1]
    ntest = X_test.shape[0] * X_test.shape[1]

    print('Train shape', X_train.shape, 'test shape', X_test.shape)
    
    if shuffle:
        print("Shuffling the data using random state", random_state)
        X_train, Y_train = shuffle(X_train, Y_train, random_state=random_state)
        
        
    return X_train.astype(out_dtype), Y_train.astype(out_dtype), X_test.astype(out_dtype), Y_test.astype(out_dtype)
