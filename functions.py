import pandas

def show_global_view(lc, i):
    # plt.figure(figsize=(20,10))
    lc.plot()
    plt.title(f'Global View: {i}', fontsize=16)
    show_inline_matplotlib_plots()

def show_local_view(lc, i):
    # plt.figure(figsize=(15,10))
    lc.fold(period=500).plot()
    plt.title(f'Folded View: {i}', fontsize=16)
    show_inline_matplotlib_plots()
    
def plot_shap_values(case_interest_lc, i):
    plt.figure(figsize=(15,10))
    # index = case_interest_lc.iloc[i].index
    dd = predicted_planets_shaps.iloc[i][1:]
    dd = dd.sort_values(ascending=False)
    dd[:10].sort_values(ascending=True).plot(kind='barh')
    plt.title(f'Shap Value Plot: {i}', fontsize=16)
    show_inline_matplotlib_plots()

def plot_selected(i):
        clear_output()
        lc = lk.LightCurve(flux=case_interest_lc.iloc[i])
        if dropdown.value == 'Global View':
            printmd(f'### Global View for index: {i}')
            show_global_view(lc, i)          
        if dropdown.value == 'Folded View':
            printmd(f'### Folded View for index: {i}')
            show_local_view(lc, i)
        if dropdown.value == 'SHAP Values':
            printmd(f'### SHAP Values for index: {i}')
            plot_shap_values(case_interest_lc, i)
            
def next_button_clicked(b):
    with out:
        global ix
        ix += 1

def prev_button_clicked(b):
    with out:
        global ix
        ix -= 1
        if ix<0: ix=0
        
def refresh_button_clicked(b):
    with out:
        global ix
        plot_selected(ix)