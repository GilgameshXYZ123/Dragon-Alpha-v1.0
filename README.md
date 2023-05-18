Thank you for downing Dragon-Alpha&cu32. 
I usually abbreviate Dragon-Alpha as Alpha.

# I.Pay-Attention
**1.** cu32 now is only complied for 64-bit Windows, so kindly run Alpha’s apps on such OS.<br>
**2.** Please make sure: the JDK version is greater than 8.0<br>
**3.** Please make sure: compute >= 61, sm >= 61<br>
**4.** Kindly read “Arxiv.pdf” first, to briefly understand Alpha.<br>
**5.** Alpha has only been executed on GTX 1050, RTX 3060ti GPU, and presently its applications can only be executed on CUDA GPU.<br>
**6.** Since I am the only-one programmer to build Alpha, I must pay my main attention to the code instead of the document, to complete Alpha’s prototype in time. If you have some questions, just see the source-code. Sorry, my personal abilities are really limited.<br> 


# II. Files
- **Arxiv.pdf**  an article talking about the background, characteristics, architecture and experiments of Alpha, preprinted on arxiv.org, at: https://arxiv.org/abs/2305.08819.<br>
- **exec**  the executable files of Alpha.
  - **lib**  Java libraries of Alpha, which are jar-files complied by Java-code.<br>
  - **native-lib**  native libraries of Dragon-Alpha. They are dynamic-linked-libraries, and integrated to Alpha at the bottom through JNI.<br>
    - **cuda_float32**  contains the executable files of cu32. Presently, cu32 has only been complied for 64-bit Windows, and will be compiled for Linux in the near future<br>
  - **icon**  Alpha’s logo. If you like it, set it for Alpha’s home directory. <br> 
  - **data**  Alphas’ built-in data-sets, including MINIST, cifar-10, cifar-100 and Soccer. Please decompress before use them.<br> 
  - **src**  the source-code of Dragon-Alpha<br>
  - **alpha_src**  the Java source-code of Alpha. You rename this directory to ‘src’ and integrate it to your own Java-project. I suggest using NetBeans to read such source-code, since I use NetBeans to build Alpha.<br>
  - **zutil_src**  ZUtil is an auxiliary library for Alpha. Since I wrote it in my sophomore year, it may have some unreasonable aspects in programming-style and architecture. I only uploaded a part of it, so kindly use ZUTIL-STD-1.1.jar instead of the source-code.<br>
  - **cu32_src**  the C++ source-code of cu32, consists of 13 Visual Studio (VS2017) projects. To open such projects on your PC, please make sure your VS can build CUDA projects. You also need to add jdk.include&jdk.lib to such projects (such as jni.h, jvm.lib). Since I use CUDA v11.3, I suggest you to use such version too.<br>
- **experiments**  the related experimental code&data related to Arxiv.pdf.<br>
  - **alpha-code**  the experimental code of Alpha, you can take it as examples, to create your own Alpha-app. Before using Alpha’s API, you must specify the home-path of Alpha, in order to load the relative native libraries.<br>
  ![image](https://github.com/GilgameshXYZ123/Dragon-Alpha/assets/65615049/2586a7d0-0226-4bae-a575-5d9e2c8bdf66)
  - **pytorch-code**  the experimental code of PyTorch.<br>
  - **experiment-data**  console output to track some metrics for both Alpha and PyTorch, in order to make a comparison.<br>
  - **test_cuda**  some related code to test Alpha&cu32. You can take it as examples of using Alpha’s operators.<br>


# Finally, 
- My name Zhang Zhiyi, and I often use Gilgamesh as my internet name.<br>
- I was born in April, 2000, majored Computer-Science in my college, and now study Pattern-Recognition after graduate.<br>
- First, let’s talk about the reason why I create Alpha instead of using PyTorch. I prefer Java to Python, but failed to find a Java-based DL framework as excellent as PyTorch, in the past few years. Also, I want to learn more about the principles and details of DL, and like implementing them to improve my abilities. So, I started to build my own Java DL framework. Dragon-Alpha can be regraded as a continuation of Dragon, which is my graduation project.<br>
- It took me about 190 days and 200,000 lines of code to build Alpha’s prototype. In such progress, I have been suffering while enjoying, and finally benefited a lot. PyTorch&cuDNN is my opponent but also my mentor. I tried to learn its advantage, and pondered how to make some breakthrough while keeping Alpha’s own characteristics. Now, the Alpha’s prototype has been completed, and the relative paper has been written.<br>
At present, Alpha is not as polished as PyTorch, but it could be good start and have a long way to reach perfection. I am grateful to all those who provided me with support and encouragement.<br>
- It’s my honour to share the source code Alpha&cu32. Sincerely, I request and need all of you to use and improve it. If you have some related good advice and achievement, please contact me at gilgamesh@mail.ustc.edu.cn.<br>

