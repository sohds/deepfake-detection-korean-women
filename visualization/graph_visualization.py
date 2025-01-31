import matplotlib.pyplot as plt

def plot_images_bar(data, color_threshold=50, high_color='green', low_color='lightgray'):
    """
    Parameters:
    - data: pandas DataFrame
    - color_threshold: 특정 기준 값 이상일 때 적용할 색상 구분 
    - high_color: 기준 값 이상일 때의 막대 색상
    - low_color: 기준 값 미만일 때의 막대 색상
    """
    percentage_non_zero = (data != 0).mean() * 100 
    sorted_data = percentage_non_zero.sort_values(ascending=False)
    
    colors = [high_color if value > color_threshold else low_color for value in sorted_data.values]
    
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_data.index, sorted_data.values, color=colors, alpha=0.7)  
    
    for idx, value in enumerate(sorted_data.values):
        plt.text(idx, value + 1, f'{value:.2f}', ha='center', va='bottom', fontsize=23)
    
    plt.xticks(ticks=range(len(sorted_data)), labels=sorted_data.index, rotation=45, ha='right', fontsize=23)
    
    for label in plt.gca().get_xticklabels():
        label.set_size(23)
    
    plt.ylim(0, sorted_data.values.max() + 10)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.show()


def plot_area_bar(df, selected_labels, color='red', figsize=(4, 6)):
    """
    Parameters:
    - df: pandas DataFrame
    - selected_labels: 시각화할 특정 레이블 리스트
    - color: 막대 그래프 색상 
    - figsize: 그래프 크기
    """
    mean_values = df['Mean Values']
    sorted_data = mean_values.sort_values(ascending=False)
    filtered_data = sorted_data.loc[selected_labels]
    
    plt.figure(figsize=figsize)
    plt.bar(filtered_data.index, filtered_data.values, color=color, alpha=0.7)
    
    for idx, value in enumerate(filtered_data.values):
        plt.text(idx, value + 1, f'{value:.2f}', ha='center', va='bottom', fontsize=23)
    
    plt.xticks(ticks=range(len(filtered_data)), labels=filtered_data.index, rotation=45, ha='right', fontsize=23)
    
    for label in plt.gca().get_xticklabels():
        label.set_size(23)
    
    plt.ylim(0, filtered_data.values.max() + 10)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.show()
