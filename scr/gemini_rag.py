import os
import re
import time
import json
from collections import deque
from datetime import datetime
from dotenv import load_dotenv

os.environ['USE_TF'] = 'NO'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

print("SIDAS - Gemini 2.0 Flash RAG Sistemi")
print("Kadına Yönelik Şiddet Destek Chatbot v2.0 (Optimize)")

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("HATA: GOOGLE_API_KEY bulunamadı!")
    print("Lütfen .env dosyasına GOOGLE_API_KEY=your_key_here ekleyin")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

FAISS_DB_PATH = "./faiss_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
KAYIT_KLASORU = "./konusmalar"
os.makedirs(KAYIT_KLASORU, exist_ok=True)

OZEL_CEVAPLAR = {
    "sen kimsin": "Ben SIDAS. Kadına yönelik şiddet, yasal haklar ve destek mekanizmaları hakkında bilgi veren bir yapay zeka asistanıyım.",
    "kimsin": "Ben SIDAS. Kadına yönelik şiddet, yasal haklar ve destek mekanizmaları hakkında bilgi veren bir yapay zeka asistanıyım.",
    "adın ne": "Adım SIDAS. Kadına yönelik şiddet konusunda size bilgi vermek için buradayım.",
    "ne yapıyorsun": "Size kadına yönelik şiddet, yasal haklarınız ve başvurabileceğiniz yerler hakkında bilgi veriyorum.",
    "nasılsın": "İyiyim, teşekkür ederim. Size kadına yönelik şiddet konusunda nasıl yardımcı olabilirim?",
    "nasilsin": "İyiyim, teşekkür ederim. Size kadına yönelik şiddet konusunda nasıl yardımcı olabilirim?",
    "merhaba": "Merhaba! Ben SIDAS. Size nasıl yardımcı olabilirim?",
    "selam": "Selam! Kadına yönelik şiddet veya yasal haklarınızla ilgili sorularınızı cevaplayabilirim.",
    "iyi günler": "İyi günler! Size nasıl yardımcı olabilirim?",
    "günaydın": "Günaydın! Size nasıl yardımcı olabilirim?",
    "teşekkürler": "Rica ederim. Başka bir konuda yardımcı olabilir miyim?",
    "tesekkurler": "Rica ederim. Başka bir konuda yardımcı olabilir miyim?",
    "sağol": "Rica ederim. Size yardımcı olmaktan mutluluk duyarım.",
    "sagol": "Rica ederim. Size yardımcı olmaktan mutluluk duyarım.",
    "hoşçakal": "Hoşça kalın. Acil durumlarda 112'yi arayabilirsiniz.",
    "hoscakal": "Hoşça kalın. Acil durumlarda 112'yi arayabilirsiniz.",
    "görüşürüz": "Görüşürüz. Acil durumlarda 112'yi arayın.",
    "gorusuruz": "Görüşürüz. Acil durumlarda 112'yi arayın.",
    "naber": "İyiyim, teşekkür ederim. Size kadına yönelik şiddet konusunda nasıl yardımcı olabilirim?",
    "ne var ne yok": "Size yardımcı olmak için buradayım. Kadına yönelik şiddet konusunda sorularınızı cevaplayabilirim."
}

KONU_DISI = {
    'genel': ['fikra', 'şaka', 'saka', 'şarkı', 'sarki', 'film', 'dizi',
              'müzik', 'muzik', 'futbol', 'spor', 'maç', 'mac'],
    'egitim': ['matematik', 'fizik', 'kimya', 'ödev', 'odev', 'ders'],
    'diger': ['hava durumu', 'yemek', 'tarif', 'tatil', 'seyahat',
              'horoskop', 'burç', 'burc', 'dağ', 'dag', 'deniz']
}

TURKIYE_ILLERI = [
    "adana", "adıyaman", "afyon", "ağrı", "aksaray", "amasya", "ankara",
    "antalya", "ardahan", "artvin", "aydın", "balıkesir", "bartın", "batman",
    "bayburt", "bilecik", "bingöl", "bitlis", "bolu", "burdur", "bursa",
    "çanakkale", "çankırı", "çorum", "denizli", "diyarbakır", "düzce",
    "edirne", "elazığ", "erzincan", "erzurum", "eskişehir", "gaziantep",
    "giresun", "gümüşhane", "hakkari", "hatay", "ığdır", "isparta",
    "istanbul", "izmir", "kahramanmaraş", "karabük", "karaman", "kars",
    "kastamonu", "kayseri", "kilis", "kırıkkale", "kırklareli", "kırşehir",
    "kocaeli", "konya", "kütahya", "malatya", "manisa", "mardin", "mersin",
    "muğla", "muş", "nevşehir", "niğde", "ordu", "osmaniye", "rize",
    "sakarya", "samsun", "şanlıurfa", "siirt", "sinop", "sivas", "şırnak",
    "tekirdağ", "tokat", "trabzon", "tunceli", "uşak", "van", "yalova",
    "yozgat", "zonguldak"
]

ACIL_DURUMLAR = {
    'kritik': [
        'şiddet gördüm', 'siddet gordum', 'şiddet görüyorum', 'siddet goruyorum',
        'dövdü', 'dovdu', 'darp edildi', 'dayak yedim', 'vurdu', 'vuruldum',
        'bıçak', 'bicak', 'silah', 'öldür', 'oldur', 'öldüreceğim', 'oldurecegim',
        'tehlikede', 'tehlikedeyim', 'can güvenliği', 'can guvenliği',
        'korkuyorum', 'çok korkuyorum', 'kaçmak istiyorum', 'kacmak istiyorum',
        'kaçmalıyım', 'kacmaliyim', 'sığınma evi acil', 'siginma evi acil',
        'eve gelmesin', 'eve geliyor', 'kapıda bekliyor', 'kapida bekliyor'
    ],
    'yuksek': [
        'tehdit ediyor', 'tehdit aldım', 'tehdit aliyorum',
        'taciz ediyor', 'taciz ediliyor', 'taciz var',
        'tecavüz', 'tecavuz', 'cinsel şiddet', 'cinsel siddet',
        'şiddet uyguluyor', 'siddet uyguluyor', 'şiddet görüyorum',
        'yaklaşma yasağı ihlal', 'yaklasma yasagi ihlal',
        'uzaklaştırma ihlal', 'uzaklastirma ihlal'
    ],
    'orta': [
        'yardım edin', 'yardim edin', 'acil yardım', 'acil yardim',
        'hemen yardım', 'hemen yardim', 'acil durum',
        'polis çağır', 'polis cagir', 'ambulans çağır', 'ambulans cagir'
    ]
}

print("\n[1/3] Retriever yükleniyor...")

try:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={'normalize_embeddings': False}
    )

    vectorstore = FAISS.load_local(
        FAISS_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    print("Retriever hazır")

except Exception as e:
    print(f"HATA: Retriever yüklenemedi - {e}")
    exit(1)

print("\n[2/3] Gemini 2.0 Flash yapılandırılıyor...")

try:
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 512,
    }

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    print("Gemini 2.0 Flash hazır")

except Exception as e:
    print(f"HATA: Gemini yapılandırılamadı - {e}")
    exit(1)

print("\n[3/3] Memory sistemi başlatılıyor...")

class EnhancedMemory:
    """Optimize edilmiş konuşma belleği"""

    def __init__(self, k=8):
        self.k = k
        self.memory = deque(maxlen=k)
        self.full_history = []
        self.summary_cache = {}

    def add(self, soru, cevap):
        entry = {
            "soru": soru,
            "cevap": cevap[:200],
            "cevap_tam": cevap,
            "zaman": datetime.now().strftime("%H:%M:%S")
        }
        self.memory.append(entry)
        self.full_history.append(entry)
        self.summary_cache.clear()

    def get_context(self, son_n=5):
        if not self.memory:
            return ""

        cache_key = f"ctx_{son_n}"
        if cache_key in self.summary_cache:
            return self.summary_cache[cache_key]

        recent = list(self.memory)[-son_n:]
        ctx_lines = ["Önceki konuşma:"]

        for item in recent:
            ctx_lines.append(f"[{item['zaman']}] Soru: {item['soru']}")
            ctx_lines.append(f"Cevap: {item['cevap']}...")

        ctx = "\n".join(ctx_lines) + "\n"
        self.summary_cache[cache_key] = ctx
        return ctx

    def is_followup(self, soru):
        if not self.memory:
            return False

        takip_patterns = [
            r'\b(var\s*m[ıi]|nerede|adres|telefon)\b',
            r'\b(peki|ya|o\s*zaman|nas[ıi]l)\b',
            r'\b(o|bu|onun|bunun|oras[ıi]|buras[ıi])\b',
            r'\b(numara|saat|gün)[sıi]?\b'
        ]

        soru_lower = soru.lower()

        if len(soru.split()) < 10:
            for pattern in takip_patterns:
                if re.search(pattern, soru_lower):
                    return True

        return False

    def enrich(self, soru):
        if not self.is_followup(soru):
            return soru

        son_sorular = list(self.memory)[-2:]
        onemli_kelimeler = set()

        for msg in son_sorular:
            words = msg["soru"].split()

            for kelime in words:
                if len(kelime) > 3:
                    if kelime[0].isupper():
                        onemli_kelimeler.add(kelime)
                    elif kelime.lower() in TURKIYE_ILLERI:
                        onemli_kelimeler.add(kelime.capitalize())
                    elif kelime.isupper() and len(kelime) > 2:
                        onemli_kelimeler.add(kelime)

        if onemli_kelimeler:
            enriched = list(onemli_kelimeler)[:3]
            return f"{' '.join(enriched)} {soru}"

        return soru

    def get_full_history(self):
        return self.full_history

    def clear(self):
        self.memory.clear()
        self.summary_cache.clear()


memory = EnhancedMemory(k=8)
print("Memory hazır")

def normalize_turkish(text):
    """Türkçe karakterleri normalize et"""
    replacements = {
        'ı': 'i', 'ğ': 'g', 'ü': 'u', 'ş': 's', 'ö': 'o', 'ç': 'c',
        'İ': 'i', 'Ğ': 'g', 'Ü': 'u', 'Ş': 's', 'Ö': 'o', 'Ç': 'c'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.lower()


def check_special_answer(soru):
    """Özel cevapları kontrol et - Optimize"""
    soru_temiz = normalize_turkish(soru.strip())
    return OZEL_CEVAPLAR.get(soru_temiz)


def is_meaningful_query(soru):
    """Anlamlı soru kontrolü"""
    # Çok kısa
    if len(soru.strip()) < 3:
        return False

    # Sadece noktalama işareti
    if all(c in '?!.,;:' for c in soru):
        return False

    # Türkçe karakter oranı kontrolü
    turkce_harfler = sum(1 for c in soru if c.isalpha())
    if turkce_harfler < len(soru) * 0.3:  # %30'dan az harf
        return False

    return True


def check_offtopic(soru):
    """Geliştirilmiş konu dışı kontrol"""
    soru_lower = soru.lower()

    # Önce konuyla ilgili anahtar kelimeleri kontrol et
    konu_icinde = ['şiddet', 'siddet', 'kades', 'şönim', 'sonim', '6284',
                   'boşanma', 'bosanma', 'nafaka', 'çocuk', 'cocuk',
                   'mahkeme', 'avukat', 'polis', 'destek', 'yardım', 'yardim',
                   'tedbir', 'koruma', 'sığınma', 'siginma', 'şiddet', 'taciz',
                   'tehdit', 'dava', 'velayet', 'uzaklaştırma', 'yaklasma']

    if any(kelime in soru_lower for kelime in konu_icinde):
        return False  # Konuyla ilgili, filtreleme

    # Sonra konu dışı kelimeleri kontrol et
    for kategori, kelimeler in KONU_DISI.items():
        for kelime in kelimeler:
            if kelime in soru_lower:
                return True

    return False


def check_emergency(soru):
    """Gelişmiş acil durum kontrolü"""
    soru_lower = soru.lower()

    # 1. KRİTİK - Kesinlikle acil durum
    for kelime in ACIL_DURUMLAR['kritik']:
        if kelime in soru_lower:
            return True

    # 2. YÜKSEK - Bağlam kontrolü
    for kelime in ACIL_DURUMLAR['yuksek']:
        if kelime in soru_lower:
            # Şu anki durum mu, geçmiş mi?
            simdiki_zaman = ['şu an', 'su an', 'şimdi', 'simdi', 'hemen', 'acil']
            if any(x in soru_lower for x in simdiki_zaman):
                return True

            # Bilgi sorusu mu? (nedir, var mı, nasıl)
            bilgi_kaliplari = ['nedir', 'ne demek', 'var mı', 'var mi', 'nasıl', 'nereden', 'kimler']
            if any(x in soru_lower for x in bilgi_kaliplari) and '?' in soru:
                return False  # Bilgi sorusu, acil değil

            return True  # Belirsizse acil say

    # 3. ORTA - Güçlü bağlam gerekli
    for kelime in ACIL_DURUMLAR['orta']:
        if kelime in soru_lower:
            # "acil yardım", "hemen yardım" gibi kombinasyonlar
            return True

    return False


def detect_city(soru):
    """Soru içinde il ismi tespit et - Optimize"""
    soru_lower = soru.lower()

    for il in TURKIYE_ILLERI:
        if il in soru_lower:
            return il.capitalize()

    return None


def clean_context(text):
    """Context'i temizle - Optimize"""
    text = re.sub(r'\[cite[^\]]*\]|\[cite_start\]|\[cite_end\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_answer(text):
    """Gemini cevabını ayıkla - Optimize"""
    # Başlıkları temizle
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)

    # Fazla satır atlamalarını temizle
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Fazla noktaları temizle
    text = re.sub(r'[.]{3,}', '.', text)

    # Kaynak referanslarını temizle - çeşitli formatlar
    text = re.sub(r'\(Kaynak\s*\d+(?:\s*,\s*\d+)*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[Kaynak\s*\d+(?:\s*,\s*\d+)*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Kaynak\s*\d+(?:\s*,\s*\d+)*\.?', '', text, flags=re.IGNORECASE)

    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def format_sources(docs):
    """Kaynak bilgilerini formatla - Optimize"""
    sources = set()

    for doc in docs[:3]:
        kaynak = os.path.basename(doc.metadata.get('source', 'Bilinmiyor'))
        kaynak = kaynak.replace('.txt', '').replace('_', ' ').title()
        sources.add(kaynak)

    return ", ".join(sources)


def save_conversation(history, kaydet_mi=True):
    """Konuşmayı kaydet"""
    if not kaydet_mi or not history:
        return None

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sidas_konusma_{timestamp}.json"
        filepath = os.path.join(KAYIT_KLASORU, filename)

        data = {
            "tarih": datetime.now().isoformat(),
            "toplam_mesaj": len(history),
            "konusma": history
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return filepath

    except Exception as e:
        print(f"Kayıt hatası: {e}")
        return None


SYSTEM_PROMPT = """Sen SIDAS adında kadına yönelik şiddet konusunda uzmanlaşmış bir destek asistanısın.

GÖREVİN:
- Kadına yönelik şiddet, yasal haklar ve destek mekanizmaları hakkında doğru bilgi vermek
- Verilen BİLGİ ve BAĞLAM'ı kullanarak cevap vermek
- Net, anlaşılır ve empatik bir dille konuşmak

KURALLAR:
1. ÖNCE BAĞLAM'a bak - önceki konuşmayı anla
2. Sonra BİLGİ'deki bilgileri kullan
3. Takip soruları akıllıca yanıtla
4. Bilgi yoksa "Bu konuda elimde bilgi bulunmuyor" de
5. Kısa ve öz cevaplar ver (2-4 cümle)
6. Tavsiye verme - sadece bilgi paylaş
7. Acil durumlarda 112'yi söyle
8. Empati göster ama profesyonel kal
9. Cevabında kaynak numarası (Kaynak 1, 2 gibi) kullanma"""

USER_PROMPT_TEMPLATE = """{baglam}

BİLGİ:
{bilgi}

SORU: {soru}

Yukarıdaki BAĞLAM ve BİLGİ'yi kullanarak soruyu kısa ve net şekilde yanıtla."""

def rag(soru, verbose=False):
    """Ana RAG fonksiyonu - Optimize"""

    start_time = time.time()

    if verbose:
        print(f"\nSoru: {soru}")

    if not soru or len(soru.strip()) < 2:
        return {
            "soru": soru,
            "cevap": "Lütfen bir soru sorun.",
            "sure": 0,
            "kaynak": None,
            "model": "Sistem"
        }

    # Anlamlılık kontrolü
    if not is_meaningful_query(soru):
        return {
            "soru": soru,
            "cevap": "Lütfen anlamlı bir soru sorun.",
            "sure": time.time() - start_time,
            "kaynak": None,
            "model": "Filtre"
        }

    ozel_cevap = check_special_answer(soru)
    if ozel_cevap:
        return {
            "soru": soru,
            "cevap": ozel_cevap,
            "sure": time.time() - start_time,
            "kaynak": "Sistem",
            "model": "Özel Cevap"
        }

    if check_emergency(soru):
        acil_mesaj = (
            "ACİL DURUM TESPİT EDİLDİ\n\n"
            "Eğer şu anda tehlike altındaysanız:\n"
            "- 112 Acil Çağrı Merkezi\n"
            "- 155 Polis İmdat\n"
            "- 183 ALO Sosyal Destek Hattı (7/24)\n\n"
            "Lütfen güvenliğinizi sağlayın."
        )

        return {
            "soru": soru,
            "cevap": acil_mesaj,
            "sure": time.time() - start_time,
            "kaynak": "Acil Protokol",
            "model": "Acil Sistem"
        }

    if check_offtopic(soru):
        return {
            "soru": soru,
            "cevap": "Üzgünüm, bu konu benim uzmanlık alanımın dışında. Ben kadına yönelik şiddet, yasal haklar ve destek mekanizmaları hakkında bilgi verebilirim.",
            "sure": time.time() - start_time,
            "kaynak": None,
            "model": "Filtre"
        }

    sehir = detect_city(soru)
    zengin_soru = memory.enrich(soru)

    try:
        docs = retriever.invoke(zengin_soru)

        if not docs:
            cevap = "Bu konuda elimde bilgi bulunmamaktadır."
            memory.add(soru, cevap)
            return {
                "soru": soru,
                "cevap": cevap,
                "sure": time.time() - start_time,
                "kaynak": None,
                "model": "Gemini 2.0 Flash"
            }

    except Exception as e:
        return {
            "soru": soru,
            "cevap": "Üzgünüm, bilgi erişiminde bir sorun oluştu.",
            "sure": time.time() - start_time,
            "kaynak": None,
            "model": "Hata"
        }

    context_parts = []
    for idx, doc in enumerate(docs[:4], 1):
        temiz = clean_context(doc.page_content)[:600]
        context_parts.append(f"[Kaynak {idx}]\n{temiz}")

    context = "\n\n".join(context_parts)
    mem_ctx = memory.get_context(son_n=5)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        baglam=mem_ctx if mem_ctx else "",
        bilgi=context,
        soru=soru
    )

    try:
        chat = model.start_chat(history=[])
        response = chat.send_message(f"{SYSTEM_PROMPT}\n\n{user_prompt}")
        cevap = extract_answer(response.text)

        if len(cevap) < 20:
            cevap = context[:300]

    except Exception as e:
        print(f"Generation hatası: {e}")
        cevap = context[:300]

    kaynaklar = format_sources(docs)

    cevap_final = f"{cevap}\n\nKaynak: {kaynaklar}\nBu bilgi yasal tavsiye değildir. Acil durumlarda 112'yi arayın."

    cevap_response = f"{cevap}\n\nBu bilgi yasal tavsiye değildir. Acil durumlarda 112'yi arayın."

    memory.add(soru, cevap)

    return {
        "soru": soru,
        "cevap": cevap_response,
        "sure": time.time() - start_time,
        "kaynak": kaynaklar,
        "model": "Gemini 2.0 Flash"
    }