# İnsan Sesi Şiddeti Ölçüm ve Analiz Modeli

Bu projede, bir ortamda bulunan insan seslerinin şiddetini ölçerek ortamın agresif mi yoksa sakin mi olduğunu belirlemeye çalışan bir derin öğrenme modeli geliştirdik. Model, güvenlik sektöründe kullanılarak kavga veya şiddet içeren durumları erken tespit edebilir ve böylece önleyici tedbirlerin alınmasına yardımcı olabilir.

## Proje Amacı

Eğittiğimiz model, herhangi bir ortamdaki insan seslerinin şiddetini analiz eder. Bu analiz sayesinde, ortamın agresif mi yoksa sakin mi olduğunu belirleyebiliriz. Bu özellik, özellikle güvenlik sektöründe büyük önem taşır; bir kavganın veya şiddet içeren bir durumun erken tespiti, hızlı müdahale ile kazaların ve kötü olayların önüne geçilmesini sağlar.

## Kullanım Alanları

- **Güvenlik Sektörü:** Model, bir kavgayı veya şiddet içeren durumu başında algılayarak güvenlik görevlilerinin hızlı müdahale etmesini sağlar. Böylece, kazaların veya kötü durumların ortaya çıkması engellenebilir.
- **Akıllı Ev Sistemleri:** Ev otomasyonu kapsamında, ses şiddeti analizi ile ev içindeki huzursuzluk durumlarını belirleyebilir.
- **Kamu Alanları:** Parklar, alışveriş merkezleri gibi kamu alanlarında güvenliği artırmak için kullanılabilir.

## Proje Dosyaları

### Learning.py

Bu dosya, modelin eğitimi için kullanılır. İçerisindeki değerleri değiştirerek modelin hassaslığını ve performansını ayarlayabilirsiniz.

### Model_Test.py

Bu dosya, eğitilen modeli test etmek için kullanılır. Modelin doğruluğunu ve güvenilirliğini bu dosya ile ölçebilirsiniz.

### Gereksinimler

- Python 3.x
- TensorFlow veya PyTorch
- NumPy
- Scikit-learn
- Matplotlib
- OS (Operating System)
- Librosa
- Seaborn
- Pickle
- Subprocess

### Kurulum

```bash
git clone https://github.com/kullanici_adi/ses-analiz-modeli.git
cd ses-analiz-modeli
pip install -r requirements.txt
