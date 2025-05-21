import matplotlib.pyplot as plt

class ProgressPlotter:
    def __init__(self):
        # Configuración de la figura
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_title('Progreso de Recompensas')
        self.ax.set_xlabel('Pasos')
        self.ax.set_ylabel('Valor')

        # Inicialización de listas para almacenar los valores
        self.steps = []
        self.rewards = []
        self.total_rewards = []

    def update_plot(self, step_count, progress_reward, total_reward):
        # Agregar los nuevos valores a las listas
        self.steps.append(step_count)
        self.total_rewards.append(total_reward)
        self.rewards.append(list(progress_reward.values()))  # Guardamos las recompensas individuales

        # Limpiamos el gráfico y lo actualizamos
        self.ax.clear()
        self.ax.set_title('Progreso de Recompensas')
        self.ax.set_xlabel('Pasos')
        self.ax.set_ylabel('Valor')

        # Graficamos las recompensas
        self.ax.plot(self.steps, self.total_rewards, label="Recompensa Total", color="blue")
        
        # Graficamos las recompensas individuales por cada clave
        for idx, key in enumerate(progress_reward.keys()):
            self.ax.plot(self.steps, [r[idx] for r in self.rewards], label=key)

        # Añadimos leyenda
        self.ax.legend()

        # Actualizamos la pantalla
        plt.draw()
        plt.pause(0.1)