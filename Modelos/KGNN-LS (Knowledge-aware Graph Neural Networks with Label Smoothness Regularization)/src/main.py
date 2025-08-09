import argparse
import numpy as np
from time import time
from data_loader import load_data
from train import train

# Importamos el recomendador basado en historial
from user_history_recommender import UserHistoryRecommender

try:
    import psutil
except ImportError:
    print("Warning: psutil module not found. System metrics tracking might be limited.")
    print("Please install using: pip install psutil")

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib module not found. Graphs will not be generated.")
    print("Please install using: pip install matplotlib")

np.random.seed(555)

parser = argparse.ArgumentParser()


# movie
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=50, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=16, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=128, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-2, help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=1.0, help='weight of LS regularization')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--track_emissions', type=bool, default=True, help='track CO2 emissions')
# Nuevos argumentos para el recomendador basado en historial
parser.add_argument('--use_history', type=bool, default=False, help='use history-based recommendation')
parser.add_argument('--history_weight', type=float, default=0.3, help='weight for history component')
parser.add_argument('--recommend_for_user', type=int, default=None, help='generate recommendations for specific user ID')
parser.add_argument('--top_k', type=int, default=10, help='number of top recommendations to show')


show_loss = False
show_time = False
show_topk = False

t = time()

args = parser.parse_args()
data = load_data(args)

# Si se solicita recomendación para un usuario específico
if args.recommend_for_user is not None:
    from user_recommendation import recommend_for_user, explain_recommendation
    
    print(f"\nGenerando recomendaciones para el usuario {args.recommend_for_user}...")
    model_path = f"./model/{args.dataset}_best.ckpt"
    
    recommendations = recommend_for_user(
        model_path, 
        data, 
        args.recommend_for_user, 
        top_k=args.top_k,
        use_history=args.use_history,
        history_weight=args.history_weight
    )
    
    print(f"\nRecomendaciones para el usuario {args.recommend_for_user}:")
    for i, (item_id, score) in enumerate(recommendations):
        # Obtener explicación
        explanation = explain_recommendation(data, args.recommend_for_user, item_id, args.history_weight)
        
        print(f"{i+1}. Item {item_id} - Score: {score:.4f}")
        
        if explanation["success"]:
            print(f"   Predicción por historial: {explanation.get('history_prediction', 'N/A')}")
            print(f"   Confianza: {explanation.get('confidence', 'N/A')}")
            
            if explanation.get("similar_items"):
                print("   Items similares en el historial del usuario:")
                for sim_item_id, sim_rating, sim_score in explanation["similar_items"][:3]:
                    print(f"     - Item {sim_item_id} - Rating: {sim_rating:.1f}, Similitud: {sim_score:.2f}")
        else:
            print(f"   {explanation['message']}")
        
        print()
else:
    # Entrenar el modelo normalmente
    results = train(args, data, show_loss, show_topk)
    
    # Si se solicita usar historial
    if args.use_history and results is not None:
        print("\n=== Aplicando recomendación basada en historial ===")
        
        # Extraer datos de test y predicciones del modelo
        test_data = data[6]
        if 'logits' in results:
            model_predictions = results['logits']
        elif 'pred_y' in results:
            model_predictions = results['pred_y']
        else:
            print("No se pudieron encontrar las predicciones del modelo en los resultados")
            model_predictions = None
        
        if model_predictions is not None:
            # Inicializar recomendador basado en historial
            history_recommender = UserHistoryRecommender(
                train_data=data[4],  # Datos de entrenamiento
                kg_data=data,        # Datos del grafo de conocimiento
                history_weight=args.history_weight
            )
            
            # Aplicar la combinación de predicciones
            enhanced_predictions = []
            for i, test_instance in enumerate(test_data):
                user_id = int(test_instance[0])
                item_id = int(test_instance[1])
                model_pred = float(model_predictions[i])
                
                # Combinar predicción del modelo con historial
                combined_pred = history_recommender.predict_rating_with_history(
                    model_pred, user_id, item_id
                )
                enhanced_predictions.append(combined_pred)
            
            # Convertir a numpy array
            enhanced_predictions = np.array(enhanced_predictions)
            
            # Calcular métricas para ambas predicciones
            from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
            
            y_true = test_data[:, 2]
            
            model_auc = roc_auc_score(y_true, model_predictions)
            model_f1 = f1_score(y_true, np.round(model_predictions))
            model_rmse = np.sqrt(mean_squared_error(y_true, model_predictions))
            
            enhanced_auc = roc_auc_score(y_true, enhanced_predictions)
            enhanced_f1 = f1_score(y_true, np.round(enhanced_predictions))
            enhanced_rmse = np.sqrt(mean_squared_error(y_true, enhanced_predictions))
            
            print("\n=== Comparación de Resultados ===")
            print(f"Modelo KGNN-LS:                AUC: {model_auc:.4f}, F1: {model_f1:.4f}, RMSE: {model_rmse:.4f}")
            print(f"Modelo KGNN-LS + Historial:    AUC: {enhanced_auc:.4f}, F1: {enhanced_f1:.4f}, RMSE: {enhanced_rmse:.4f}")

if show_time:
    print('time used: %d s' % (time() - t))
    
# PARA EJECUTAR: python main.py --use_history True --history_weight 0.3 