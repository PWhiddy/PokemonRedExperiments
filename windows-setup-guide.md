## Windows guide

Made by discord user @Aisiktir

1.) Python, https://www.python.org/ftp/python/3.11.6/python-3.11.6-amd64.exe 
 
2.) Install it, make sure "Add Python to path" is enabled 

3.) Git, https://github.com/git-for-windows/git/releases/download/v2.42.0.windows.2/Git-2.42.0.2-64-bit.exe 
 
4.) Install it  

4.1)

![https://i.postimg.cc/DyJ1m4YG/image.png](https://i.postimg.cc/DyJ1m4YG/image.png)

4.2)

![https://i.postimg.cc/C5WBbJ6q/image.png](https://i.postimg.cc/C5WBbJ6q/image.png)

4.3)

![https://i.postimg.cc/hPd7KWfb/image.png](https://i.postimg.cc/hPd7KWfb/image.png)

4.4)

![https://i.postimg.cc/LsNJxmbR/image.png](https://i.postimg.cc/LsNJxmbR/image.png)

4.5)

![https://i.postimg.cc/1XN4Rw55/image.png](https://i.postimg.cc/1XN4Rw55/image.png)

4.6)

![https://i.postimg.cc/HjDnrnV0/image.png](https://i.postimg.cc/HjDnrnV0/image.png)

4.7)

![https://i.postimg.cc/W17mPLHY/image.png](https://i.postimg.cc/W17mPLHY/image.png)

4.8)

![https://i.postimg.cc/cJ9Qr7ZQ/image.png](https://i.postimg.cc/cJ9Qr7ZQ/image.png)

4.9)

![https://i.postimg.cc/638CR75F/image.png](https://i.postimg.cc/638CR75F/image.png)

4.10)

![https://i.postimg.cc/2jB4SBtZ/image.png](https://i.postimg.cc/2jB4SBtZ/image.png)

4.11)

![https://i.postimg.cc/sD4GMP4v/image.png](https://i.postimg.cc/sD4GMP4v/image.png)

4.12)

![https://i.postimg.cc/bvWs7mpF/image.png](https://i.postimg.cc/bvWs7mpF/image.png)


5.) Microsoft C++ Build Tools, https://visualstudio.microsoft.com/visual-cpp-build-tools/ 

6.) Install it
![https://i.postimg.cc/Yq15fqds/image.png](https://i.postimg.cc/Yq15fqds/image.png)
 
Windows 11
![https://i.postimg.cc/VkKHDSnD/image.png](https://i.postimg.cc/VkKHDSnD/image.png)
 
Windows 10
![https://i.postimg.cc/43KwPWJx/image.png](https://i.postimg.cc/43KwPWJx/image.png)

7.) Open cmd 
 
7.1) Browse to the path of the git, for example "cd C:\Users\YourWindowsUsername\Downloads\PokemonRedExperiments-master\baselines"
 
7.2) Type "pip install -r requirements.txt"
 
7.3) If you didn't got any warnings or red text you're good to go and use it
 

If you had already a version of python installed, Python 3.11.6 might not work from the above command lines, you should use the below paths for the pip command and the python command

"%localappdata%\Programs\Python\Python311\Scripts\pip.exe" install -r requirements.txt

"%localappdata%\Programs\Python\Python311\python.exe" run_pretrained_interactive.py

"%localappdata%\Programs\Python\Python311\python.exe" run_baseline_parallel_fast.py
