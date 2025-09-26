# Akbank-deep-learning-project
Akbank Deep Learning Bootcamp
# Projenin Amacı  
Bu çalışmanın temel amacı, Kaggle platformundan elde edilen **deri hastalıkları görsel veri seti** kullanılarak özel olarak tasarlanmış bir **Evrişimsel Sinir Ağı (Custom CNN)** ve **transfer öğrenme tabanlı MobileNetV2** mimarileri ile sınıflandırma modelleri geliştirmektir.  
Deri hastalıklarının **erken ve doğru biçimde teşhis edilmesi**, hastaların yaşam kalitesini artırmak, tedavi süreçlerini hızlandırmak ve sağlık uzmanlarının iş yükünü azaltmak açısından kritik öneme sahiptir. Bu bağlamda, yapay zekâ tabanlı bilgisayar destekli tanı sistemlerinin geliştirilmesi, dermatoloji alanında teşhisin doğruluğunu artıracak yenilikçi çözümler sunma potansiyeline sahiptir.  
Çalışma kapsamında:  
- Görsel verilerin ön işleme adımlarından geçirilmesi,  
- Eğitim ve doğrulama veri kümelerinin oluşturulması,  
- **Custom CNN modeli ile sıfırdan bir sınıflandırıcı geliştirilmesi ve hiperparametre optimizasyonunun yapılması**,  
- **MobileNetV2 modeli ile transfer öğrenme yaklaşımının uygulanması**  
amaçlanmaktadır.  
Elde edilen modellerin performansları karşılaştırmalı olarak incelenerek, **en uygun mimarinin belirlenmesi** hedeflenmektedir. Bu araştırmadan elde edilecek bulguların, yapay zekâ tabanlı tıbbi görüntü analizi çalışmalarına katkı sağlaması ve dermatolojik tanı süreçlerinde kullanılabilecek güvenilir sınıflandırma sistemlerinin geliştirilmesine zemin oluşturması beklenmektedir.

# Veri Seti Hakkında Bilgi  
Bu çalışmada kullanılan veri seti, Kaggle platformunda yayımlanmış olan **DermNet Dermatoloji Görsel Veri Seti**’dir. Veri seti, dermatolojik hastalıkların tanısına yönelik olarak:  
- **23 sınıf**,  
- **toplam 19.559 görsel**  
içermektedir.  
Her sınıf, ilgili hastalığa ait farklı sayıda görsel barındırmaktadır. Örneğin:  
- *Seborrheic Keratoses and other Benign Tumors* sınıfında **1371 görüntü**,  
- *Urticaria Hives* sınıfında yalnızca **212 görüntü** bulunmaktadır.  
Bu durum, **sınıflar arasında veri dengesizliği (class imbalance)** problemini ortaya çıkarmakta ve model geliştirme sürecinde özel önlemler alınmasını gerektirmektedir.  
Veri seti iki ana alt kümeye ayrılmıştır:  
- **Eğitim (train)** kümesi: her sınıf için yüzlerce örnek, toplamda birkaç bini aşan görsel içerir.  
- **Test (test)** kümesi: daha az sayıda görsel içerir ve modellerin performansını değerlendirmek amacıyla kullanılır.  
Görseller farklı çözünürlüklerde olup çeşitli ışık koşulları, açılar ve deri bölgeleri üzerinden elde edilmiştir. Bu çeşitlilik, geliştirilen modelin gerçek dünyadaki farklı senaryolara daha iyi **genelleme yapabilmesini** sağlamaktadır.  
Veri seti, klinik dermatoloji uygulamalarına uygun, çok sınıflı ve gerçek hayattaki çeşitliliği yansıtan bir yapı sunmaktadır. Bu nedenle, derin öğrenme tabanlı sınıflandırma modellerinin geliştirilmesi için oldukça elverişli bir kaynak niteliğindedir.

# Kullanılan Yöntemler  
Bu çalışmada deri hastalıkları görsellerinin sınıflandırılması amacıyla iki farklı derin öğrenme yaklaşımı kullanılmıştır: özel olarak tasarlanmış **Evrişimsel Sinir Ağı (Custom CNN)** ve **transfer öğrenme tabanlı MobileNetV2**.  
## 1. Custom CNN Mimarisi  
Projede geliştirilen özel CNN modeli, klasik evrişimsel katmanların yanı sıra **SeparableConv2D blokları**, **residual (skip) bağlantılar**, **Batch Normalization (BN)**, **Dropout** ve **L2 düzenlileştirme** teknikleriyle zenginleştirilmiştir.  
- **SeparableConv2D blokları:** Parametre sayısını ve hesaplama maliyetini azaltırken, özellik çıkarımını etkin biçimde sürdürmektedir.  
- **Residual bağlantılar:** Derin ağlarda görülen gradyan sönmesi problemini azaltarak daha kararlı bir öğrenme süreci sağlar.  
- **Batch Normalization:** Aktivasyonların normalleştirilmesini sağlayarak daha hızlı yakınsama ve daha iyi genelleme sunar.  
- **Dropout:** Aşırı öğrenmeyi (overfitting) önlemek amacıyla farklı oranlarda uygulanmıştır.  
- **L2 düzenlileştirme:** Parametre büyüklüklerini kontrol ederek genelleme performansını artırır.  
Ağın mimarisi:  
- **Conv2D tabanlı stem katmanı**,  
- Artan filtre boyutlarına sahip **3 adet SeparableConv2D bloğu**,  
- **Global Average Pooling** + yoğun katmanlar,  
- Çıkışta **softmax aktivasyon fonksiyonu**.  
Eğitim sürecinde:  
- **Adam optimizer**  
- **Label smoothing içeren Categorical Crossentropy kaybı** kullanılmıştır.  
Ayrıca, **class weight yöntemi** ile az sayıda örneğe sahip sınıfların model üzerindeki etkisi artırılmıştır.

## 2. Hiperparametre Optimizasyonu (Custom CNN için)  
Custom CNN mimarisinin performansını artırmak ve overfitting/underfitting sorunlarını azaltmak için **hiperparametre optimizasyonu** yapılmıştır.  
- **Keras Tuner** kütüphanesi ile **Bayesian Optimization** yöntemi kullanılmıştır.  
- Optimize edilen hiperparametreler:  
  - Katman sayısı (2–4 blok)  
  - Filtre sayıları (32–256)  
  - Kernel boyutları (3×3, 5×5)  
  - Dropout oranları (0.0–0.6)  
  - Dense katman boyutları (256–512 nöron)  
  - L2 düzenlileştirme katsayısı  
  - Öğrenme oranı (1e-3, 5e-4, 1e-4)  
  - Optimizer seçimi (Adam, RMSprop, SGD)  
  - Batch size (16, 32)  
  - Label smoothing (0.0, 0.05, 0.1)  
Her deneme için model eğitilmiş, **val_accuracy** başarı kriteri olarak kullanılmıştır.  
**EarlyStopping** ve **ReduceLROnPlateau** callback’leriyle eğitim süreci optimize edilmiştir.

## 3. MobileNetV2 ile Transfer Öğrenme  
- MobileNetV2, literatürde yaygın kullanılan, gömülü cihazlarda verimli çalışabilen bir mimaridir.  
- **Depthwise separable convolutions** ve **inverted residual blokları** sayesinde hem parametre verimliliği hem yüksek doğruluk sağlar.  
- Bu çalışmada, **ImageNet üzerinde önceden eğitilmiş ağırlıklar** kullanılmıştır.  
- Son katmanlar çıkarılarak dermatolojik veri setine uygun yeni sınıflandırıcı eklenmiştir.  
Böylece, büyük ölçekli veri setinden öğrenilen genellemeler küçük ölçekli bu veri setine aktarılmıştır.  

## 4. Veri Ön İşleme ve Artırma  
Modelin eğitiminde aşırı öğrenmeyi önlemek ve çeşitliliği artırmak için veri artırma (data augmentation) teknikleri uygulanmıştır:  
- Rastgele yatay çevirme  
- Rastgele döndürme  
- Yakınlaştırma (zoom)  
- Kontrast değişiklikleri  
Ayrıca:  
- Tüm görseller **224×224 boyutuna ölçeklenmiş**,  
- Pikseller **[0,1] aralığına normalize edilmiş**,  
- Eğitim, doğrulama ve test veri kümeleri **caching** ve **prefetching** teknikleriyle hızlandırılmıştır.


# Sonuçlar

## Custom CNN
Bu çalışmada geliştirilen **Custom CNN** modeli, deri hastalıkları veri seti üzerinde düşük başarı göstermiştir.  
- Eğitim doğruluğu: %8–16  
- Doğrulama doğruluğu: en fazla %15  
- Test doğruluğu: **%2.45**  
Çoğu sınıf için precision, recall ve F1-score değerleri 0.0 çıkarken, sadece birkaç sınıfta sınırlı başarı sağlanabilmiştir.  
Bu sonuçlar, çok sınıflı ve dengesiz veri setlerinde Custom CNN mimarisinin yetersiz kaldığını göstermektedir.  
**Hiperparametre Optimizasyonu:**  
Custom CNN üzerinde **Keras Tuner (Bayesian Optimization)** ile hiperparametre optimizasyonu yapılmıştır.  
Amaç, doğrulama başarımını artırmak ve aşırı öğrenmeyi (overfitting) azaltmaktır.  

---

## MobileNetV2
Bu çalışmada **transfer learning** yaklaşımıyla **MobileNetV2** modeli kullanılmıştır. Model ImageNet üzerinde önceden eğitilmiş ağırlıklarla başlatılmış ve iki aşamalı eğitim yapılmıştır:  
- **Stage 1 (üst katman eğitimi):** MobileNetV2’nin taban katmanları dondurulmuş, sadece eklenen dense katmanlar eğitilmiştir.  
- **Stage 2 (fine-tuning):** MobileNetV2’nin son 30 katmanı serbest bırakılmış ve düşük öğrenme oranıyla tekrar eğitilmiştir.  
**Sonuçlar:**  
- Eğitim/Doğrulama doğruluğu: %70–80 arası  
- Test doğruluğu: **%75–80**  
- Precision, recall ve F1-score değerleri çoğu sınıfta dengeli ve yüksektir.  
Bu bulgular, **transfer learning’in küçük ve dengesiz veri setlerinde Custom CNN’e göre çok daha başarılı** olduğunu göstermektedir.  



