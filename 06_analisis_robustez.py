"""Diagnósticos post-hoc del procedimiento congelado; nunca reseleccionan modelos.

Ejecutar después de 09. Compara periodos y territorios, sensibilidad de la
selección a los cuatro orígenes y discontinuidades de la trayectoria final.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from forecasting.experiment import ROOT, REPORTS, SELECTION_PATH
from forecasting.unified import metric_table


def selection_sensitivity(detail, chosen):
    rows=[]
    for horizon, group in detail.groupby('horizon'):
        selected=chosen[str(int(horizon))]
        origins=sorted(group.origin.unique())
        for omitted in origins:
            remaining=group.loc[group.origin.ne(omitted)]
            ranks=remaining.groupby('candidate_id').absolute_error.mean().sort_values(kind='stable')
            best=ranks.index[0]
            rows.append({'horizon':horizon,'omitted_origin':omitted,
                         'selected_candidate':selected,'sensitivity_best_candidate':best,
                         'selection_unchanged':best==selected,
                         'selected_remaining_MAE':ranks[selected],
                         'best_remaining_MAE':ranks.iloc[0],
                         'role':'descriptive_only_no_reselection'})
    return pd.DataFrame(rows)


def trajectory_diagnostics(forecast):
    rows=[]
    for code, part in forecast.groupby('province_code'):
        part=part.sort_values('horizon')
        changes=part.prediccion.pct_change()*100
        rows.append({'province_code':str(code).zfill(2),'province':part.provincia.iloc[0],
                     'max_absolute_quarterly_change_pct':changes.abs().max(),
                     'direction_reversals':int((np.sign(changes.dropna()).diff().abs()>0).sum()),
                     'different_candidates':part.candidate_id.nunique(),
                     'note':'independent_horizon_models_not_smoothed'})
    return pd.DataFrame(rows)


def plots(metrics, coverage):
    figures=REPORTS/'figures';figures.mkdir(parents=True,exist_ok=True)
    labels={'selected':'Selección congelada','persistence':'Persistencia',
            'annual_trend':'Tendencia interanual','holt':'Holt amortiguado',
            'recursive_huber':'Huber recursivo','damped_trend':'Tendencia amortiguada'}
    fig,ax=plt.subplots(figsize=(10,5))
    for role,part in metrics.groupby('role'):
        ax.plot(part.horizon,part.MAE,marker='o',label=labels.get(role,role))
    ax.set(xlabel='Horizonte (trimestres)',ylabel='MAE (€/m²)',title='Evaluación retrospectiva: mismos casos por modelo y horizonte')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(figures/'evaluation_mae.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for confidence,part in coverage.groupby('nominal_coverage'):
        line,=ax.plot(part.horizon,part.empirical_coverage*100,marker='o',label=f'Cobertura observada ({confidence:.0%} nominal)')
        ax.axhline(confidence*100,color=line.get_color(),linestyle='--',alpha=.7)
    ax.set(xlabel='Horizonte (trimestres)',ylabel='Cobertura (%)',ylim=(0,102),title='Bandas calibradas en 2019–2022 y evaluadas desde 2023')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(figures/'interval_coverage.png',dpi=160);plt.close(fig)


def main():
    selection=json.loads(SELECTION_PATH.read_text(encoding='utf-8'))
    from forecasting.direct import ModelSpec
    chosen={h:ModelSpec.from_dict(spec).key for h,spec in selection['selected_by_horizon'].items()}
    development=pd.read_parquet(REPORTS/'selection_predictions.parquet')
    sensitivity=selection_sensitivity(development,chosen)
    sensitivity.to_csv(REPORTS/'robustness_leave_one_origin_out.csv',index=False)
    sensitivity.groupby('horizon',as_index=False).agg(
        unchanged_fraction=('selection_unchanged','mean'),
        comparisons=('omitted_origin','count')).to_csv(REPORTS/'robustness_selection_stability.csv',index=False)
    calibration=pd.read_parquet(REPORTS/'calibration_predictions.parquet')
    evaluation=pd.read_parquet(REPORTS/'evaluation_predictions.parquet')
    selected_eval=evaluation.loc[evaluation.role.eq('selected')]
    combined=pd.concat([calibration.assign(phase='calibration'),selected_eval.assign(phase='evaluation')],ignore_index=True)
    metric_table(combined,['phase','horizon']).to_csv(REPORTS/'robustness_temporal.csv',index=False)
    metric_table(selected_eval,['province_code','horizon']).to_csv(REPORTS/'robustness_territorial.csv',index=False)
    forecast=pd.read_csv(ROOT/'data/forecasts/prevision_provincial_2026T2_2028T4.csv',dtype={'province_code':str})
    trajectory_diagnostics(forecast).to_csv(REPORTS/'forecast_trajectory_diagnostics.csv',index=False)
    plots(pd.read_csv(REPORTS/'evaluation_metrics.csv'),pd.read_csv(REPORTS/'interval_coverage.csv'))
    print('[ROBUSTNESS] Descriptive diagnostics saved; selection unchanged.')


if __name__=='__main__':
    main()
