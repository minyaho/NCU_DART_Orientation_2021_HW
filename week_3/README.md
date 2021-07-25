# Orientation 2021: Homework Week 3

# Docker 的基礎使用:
## 基礎指令訓練:
### 1. 執行 Docker Container
以 `docker run` 透過指定 image 來建立 Container，
```bash
sudo docker run -itd --name ubuntu ubuntu:16.04 /bin/bash
```
> -i , --interactive : 讓 Container 的標準輸入保持打開
> -t , --tty : 讓Docker分配一個虛擬終端（pseudo-tty）並綁定到 Container 的標準輸入上
> -d , --detach : 讓 Container 處於背景執行狀態並印出 Container ID
> --name : 指定 Container 名稱
> -p , --publish : 將 Container 發布到指定的port號
### 2. 啟動 Docker Container
```bash
docker start ubuntu
```
### 3. 重啟 Docker Container
```bash
docker restart ubuntu
```
### 4. 停止 Docker Container
```bash
sudo docker stop ubuntu
```
### 5. 進入 Docker Container
有兩種方式，效果各不同:
- 方法一

    ```bash
    docker exec -it ubuntu bash
    ```
    
    docker exec 指令可以開啟多個終端。當以 exec 進入 Container 輸入 exit 離開， Container 本身依然在背景執行，並不會被關閉。
- 方法二

    ```bash
    docker attach ubuntu
    ```
    
    docker attach 指令為開啟一個正在運行的終端，使用上較不方便。當以 attach 進入 Container 輸入 exit 離開， Container 本身也隨之關閉。如果按下 Ctrl+P 加 Ctrl+Q 的方式離開， Container 會依然在背景狀態執行。
### 6. 刪除 Docker Container
```bash
docker rm ubuntu
```
### 7. 刪除正在進行 Docker Container
```bash
docker kill ubuntu
```
### 8. 匯出 Container
```bash
docker export ubuntu > ubuntu.tar
```
### 9. 匯入 Container
```bash
cat ubuntu.tar | sudo docker import - ubuntu:16.04
```
### 10. 暫停 Container
```bash
docker pause DOCKER_ID
```
若要讓暫停的 Docker 容器恢復執行，則使用：
```bash
docker unpause DOCKER_ID
```

</br>

# 作業流程:
1. 設定好VPN後，用 ssh 登入到 server。
   ```shell
   ssh username@{server.ip} -p {port}
   ```
   
   - 結果
   
   ![](https://i.imgur.com/Co60H6R.png)
   
   
2. 使用 bash 當作接下來的 shell。(一般使用者預設 sh 為 dash)
    ```shell
    bash
    ```
    
    - 結果
    
    ![](https://i.imgur.com/14wuhud.png)
    

3. 創建一個 container
    ```shell
    docker run --name {container.name} -p <對外port號>:<container內部port號> -it {image.name}
    ```
    
    - 結果
    
    ![](https://i.imgur.com/bhbOahM.png)
    

4. cd 至 home 目錄
    ```shell
    cd home
    ```
    - 結果
    
    ![](https://i.imgur.com/UcWC6Lb.png)
    

5. 安裝 vim
    ``` shell
    apt-get install vim
    ```

6. 撰寫 python code 並保存
    - 結果
    
    ![](https://i.imgur.com/kjX5ld6.png)
    

7. 運行 code
    - 結果
    
    ![](https://i.imgur.com/Yjb8hDd.png)
    

# 程式碼: mnist 手寫數字辨識
```python=
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout
import numpy as np
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# Reshape dataset
x_train = x_train.reshape(len(x_train),784).astype('float32')
x_test = x_test.reshape(len(x_test),784).astype('float32')

# Normalize
x_train = x_train/255
x_test = x_test/255

# One-Hot Encoding
y_train_onehot = np_utils.to_categorical(y_train)
y_test_onehot = np_utils.to_categorical(y_test)

# Buliding Model
model = Sequential()
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=128, kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training Model
train_history = model.fit(x = x_train, y = y_train_onehot, validation_split=0.2, epochs=10, batch_size=32, verbose=1)

# Accuracy score
scores = model.evaluate(x_test, y_test_onehot)
print('\naccuracy: {}'.format(scores[1])) 
```

# 參考:
1. [Day 5 關於 Container 的那些大小事](https://ithelp.ithome.com.tw/articles/10193534)
2. [Docker 常用指令與容器操作教學](https://blog.gtwang.org/linux/docker-commands-and-container-management-tutorial/)
