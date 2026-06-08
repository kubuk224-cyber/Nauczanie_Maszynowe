import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================
# 1. KONFIGURACJA
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BODY_MODEL_PATH = os.path.join(BASE_DIR, 'best_guitar_classifier.pth')
BRIDGE_MODEL_PATH = os.path.join(BASE_DIR, 'best_bridge_classifier.pth')

BODY_CLASSES = ['single_cut', 'double_cut']
BRIDGE_CLASSES = ['Floyd_style', 'Hardtail_style', 'Tremolo_style', 'Tune_o_matic_style']

# NAPRAWIONE TRANSFORMACJE: Zmuszamy zdjęcie do kwadratu 600x600 (bez wycinania środka!)
transform = transforms.Compose([
    transforms.Resize((600, 600)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 2. FUNKCJE ŁADUJĄCE MODELE
# ==========================================
def load_body_model():
    """Ładuje model ResNet-18 do rozpoznawania kształtu"""
    print("Ładowanie modelu kształtu (ResNet-18)...")
    
    # Kształty były trenowane na mniejszym ResNet-18
    model = models.resnet18(weights=None) 
    num_ftrs = model.fc.in_features # Tutaj będzie 512
    
    # Odtwarzamy architekturę z Basic.py
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, len(BODY_CLASSES))
    )
    
    model.load_state_dict(torch.load(BODY_MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval() 
    return model

def load_bridge_model():
    """Ładuje model ResNet-50 do rozpoznawania mostków"""
    print("Ładowanie modelu mostków (ResNet-50)...")
    
    # Mostki były trenowane na potężniejszym ResNet-50
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features # Tutaj będzie 2048
    
    # Odtwarzamy architekturę z trenowania mostków
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, len(BRIDGE_CLASSES))
    )
    
    model.load_state_dict(torch.load(BRIDGE_MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model
# ==========================================
# 3. GŁÓWNA FUNKCJA ANALIZUJĄCA I RYSUJĄCA
# ==========================================
def analyze_guitar(image_path, body_model, bridge_model):
    print(f"\nAnalizowanie zdjęcia: {image_path}")
    
    try:
        original_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Błąd! Nie można otworzyć pliku {image_path}. Upewnij się, że plik istnieje.")
        return

    # Przygotowanie tensora
    input_tensor = transform(original_image)
    input_batch = input_tensor.unsqueeze(0).to(DEVICE) 

    with torch.no_grad():
        # 1. Kształt (Body)
        body_out = body_model(input_batch)
        body_probs = torch.nn.functional.softmax(body_out[0], dim=0)
        body_confidence, body_pred_idx = torch.max(body_probs, 0)
        predicted_body = BODY_CLASSES[body_pred_idx.item()]
        
        # 2. Mostek (Bridge)
        bridge_out = bridge_model(input_batch)
        bridge_probs = torch.nn.functional.softmax(bridge_out[0], dim=0)
        bridge_confidence, bridge_pred_idx = torch.max(bridge_probs, 0)
        predicted_bridge = BRIDGE_CLASSES[bridge_pred_idx.item()]

    body_text = f"Kształt: {predicted_body} ({body_confidence.item()*100:.1f}%)"
    bridge_text = f"Mostek: {predicted_bridge} ({bridge_confidence.item()*100:.1f}%)"
    
    print(f"-> {body_text}")
    print(f"-> {bridge_text}")

    # ==========================================
    # 4. GENEROWANIE GRAFIKI
    # ==========================================
    plt.figure(figsize=(8, 10))
    
    # Rysujemy oryginał, ale żeby pokazać jak widzi to sieć, możemy lekko wyrównać proporcje okna
    plt.imshow(original_image)
    plt.axis('off') 
    
    text_bg_color = 'black'
    text_color = 'white'
    
    plt.title("Analiza Gitary AI\n", fontsize=18, fontweight='bold')
    
    plt.figtext(0.5, 0.15, body_text, ha="center", fontsize=16, 
                bbox=dict(facecolor=text_bg_color, alpha=0.8, edgecolor='none', pad=10),
                color=text_color, fontweight='bold')
                
    plt.figtext(0.5, 0.08, bridge_text, ha="center", fontsize=16, 
                bbox=dict(facecolor=text_bg_color, alpha=0.8, edgecolor='none', pad=10),
                color=text_color, fontweight='bold')

    output_filename = 'analiza_wynik.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nWygenerowano grafikę: {output_filename}")
    plt.show()

if __name__ == '__main__':
    model_ksztaltu = load_body_model()
    model_mostkow = load_bridge_model()
    
    # PODAJ ŚCIEŻKĘ DO ZDJĘCIA KTÓRE CHCESZ PRZETESTOWAĆ:
    testowe_zdjecie = 'gitara.jpg' # Podmień na własną nazwę pliku
    
    analyze_guitar(testowe_zdjecie, model_ksztaltu, model_mostkow)