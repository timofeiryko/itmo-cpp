import seaborn as sns
import matplotlib.pyplot as plt

def plot_sequence_validity(
    df,
    sequence_col='standard_sequence',
    figsize=(6, 4),
    palette='RdYlGn',
    title='Sequence Validity'
):
    """
    Plots distribution of valid vs invalid sequences.
    """
    valid = df[sequence_col].notna()
    
    plt.figure(figsize=figsize)
    ax = sns.countplot(
        x=valid,
        hue=valid,  # Add hue mapping
        palette=palette,
        legend=False  # Disable automatic legend
    )
    
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height}\n({height/total:.1%})",
            (p.get_x() + p.get_width()/2., height),
            ha='center', 
            va='center',
            fontsize=12,
            color='black'
        )
    
    plt.title(title)
    plt.xlabel('Validity')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Invalid', 'Valid'])
    sns.despine()
    
    return ax

def plot_invalid_categories(
    df,
    sequence_col='standard_sequence',
    category_col='sequence_category',
    figsize=(8, 6),
    palette='Reds_r',
    title='Invalid Sequences by Category'
):
    """
    Shows distribution of categories for invalid sequences.
    """
    invalid_df = df[df[sequence_col].isna()]
    order = invalid_df[category_col].value_counts().index
    
    plt.figure(figsize=figsize)
    ax = sns.countplot(x=category_col, data=invalid_df, order=order, palette=palette)
    
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    sns.despine()
    
    total = len(invalid_df)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height}\n({height/total:.1%})",
            (p.get_x() + p.get_width()/2., height),
            ha='center', va='bottom'
        )
    
    return ax