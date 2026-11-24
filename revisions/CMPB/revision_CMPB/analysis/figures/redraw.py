"""Render S4 and S5 from the locked edge inventory and component-wide annotations."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "InterGATE/results_v8"
TABLES = RESULTS / "10_manuscript_figures/tables"
OUT = HERE.parents[1] / "manuscript/figures"
COLORS = ["#af3d48", "#df811b", "#7661a0", "#c4c9ca"]
CATEGORIES = ["Basal-like signature and Hallmark", "Basal-like signature only", "Hallmark only", "Neither"]
EDGE_COLORS = ["#a1a6a7", "#348659", "#c44943"]


def layout_components(graph, completed):
    groups = sorted(nx.connected_components(graph.to_undirected()), key=lambda c:(-len(c),min(c)))
    positions = {}
    if completed:
        slots = [(0.5,0.55,0.46,0.42),(.18,.068,.055,.025),
                 (.5,.068,.055,.025),(.82,.068,.055,.025)]
    else:
        # Largest components occupy the top rows, the dyads form the final rows.
        widths = [2,5,8,10,10,10,10,10,10]
        heights = np.array([.28,.16,.12,.078,.074,.074,.074,.07,.07])
        slots = []
        top = 1.
        for n,h in zip(widths,heights):
            slots.extend([((i+.5)/n,top-h/2,.44/n,.43*h) for i in range(n)])
            top -= h
    assert len(slots) == len(groups)
    for i, (nodes, (x,y,rx,ry)) in enumerate(zip(groups,slots)):
        names = sorted(nodes)
        sub = graph.subgraph(names).to_undirected()
        if len(names)==2:
            local = np.array([[-.75,0],[.75,0]])
        else:
            xy = nx.spring_layout(sub,seed=110+i,iterations=350,method="force")
            local = np.array([xy[name] for name in names])
            local -= (local.min(0)+local.max(0))/2
            local /= np.maximum(np.max(np.abs(local),0),1e-6)
        for name, pt in zip(names,local):
            positions[name] = np.array([x,y])+np.array([rx,ry])*pt
    return positions, groups


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    comp = pd.read_csv(TABLES / "basal_strict_5of5_components.csv")
    sig = set(";".join(comp.basal_signature_genes.dropna()).split(";"))
    inventory = pd.read_excel(next((ROOT/"2027_InterGATE/Supplementary Data").glob("Supplementary_Data_1*")),sheet_name="Core_genes",header=3)
    assert sig == set(inventory.loc[inventory.basal_signature_gene.eq(True),"gene_symbol"])
    terms = pd.read_csv(RESULTS / "08c_xgboost_intergate_biology/prespecified_basal_programmes.csv")
    hits = terms[(terms.panel == "InterGATE_strict_5of5") & (terms.collection == "Hallmark")]
    hallmark = set(";".join(hits.overlap_genes).split(";"))
    panel = pd.read_csv(RESULTS/"04_controls/locked_biology_score_w0075/biology_consensus/biology_consensus_Basal-like.csv")
    edges = panel.loc[panel.selection_count.eq(5)]
    completion = pd.read_csv(TABLES/"basal_prior_completion/basal_prior_connector_edges.csv")
    info = json.loads((TABLES/"basal_prior_completion/basal_prior_completion_manifest.json").read_text())
    core = nx.from_pandas_edgelist(edges,"source_gene","target_gene","edge_type",create_using=nx.DiGraph)
    assert (len(core),core.number_of_edges())==(288,299)
    cats=[sorted(set(core)&sig&hallmark), sorted((set(core)&sig)-hallmark),
          sorted((set(core)&hallmark)-sig), sorted(set(core)-sig-hallmark)]
    assert [len(c) for c in cats]==[16,55,45,172]
    for completed in (False,True):
        graph=core.copy()
        if completed:
            graph.add_edges_from(zip(completion.source_gene,completion.target_gene))
        pos, groups=layout_components(graph,completed)
        fig,ax=plt.subplots(figsize=(12,9.5))
        fig.subplots_adjust(left=.03,right=.97,bottom=.115,top=.925)
        ax.set(xlim=(-.02,1.02),ylim=(-.02,1.02)); ax.axis("off")
        if completed:
            nx.draw_networkx_edges(graph,pos,edgelist=list(zip(completion.source_gene,completion.target_gene)),
                ax=ax,edge_color="#3678aa",style="dashed",width=.8,alpha=.8,arrows=True,arrowsize=7,node_size=21)
        for kind in range(3):
            rows = edges[edges.edge_type==kind]
            nx.draw_networkx_edges(core,pos,edgelist=list(zip(rows.source_gene,rows.target_gene)),ax=ax,
                edge_color=EDGE_COLORS[kind],width=.95,alpha=.82,arrows=True,arrowsize=7,node_size=24,
                connectionstyle="arc3,rad=.07")
        for names,col in zip(cats,COLORS):
            nx.draw_networkx_nodes(core,pos,nodelist=names,node_color=col,node_size=30,linewidths=.45,
                                   edgecolors="white",ax=ax)
        if completed:
            extra=sorted(set(graph)-set(core))
            assert set(extra)==set(info["connector_genes"])
            nx.draw_networkx_nodes(graph,pos,nodelist=extra,node_shape="D",node_color="#3678aa",
                                   node_size=43,linewidths=.5,edgecolors="white",ax=ax)
            for group in groups[1:]:
                center=np.mean([pos[name] for name in group],axis=0)
                ax.text(center[0],center[1]-.029," – ".join(sorted(group)),ha="center",va="top",fontsize=8)
        else:
            for group in groups[:2]:
                xy=np.array([pos[name] for name in group])
                ax.text(float(xy[:,0].mean()),.997,f"{len(group)} genes",ha="center",va="top",fontsize=9)
        # Label landmarks in the two largest components; identify each added connector in S5.
        labels = (info["connector_genes"] if completed else ["VGLL1","TEAD4","E2F3","LMO4","EGFR","SOX10","CDK1"])
        chosen = []
        for gene in labels:
            base = pos[gene]
            offsets = [(dx,dy) for dy in (.023,-.023,.048,-.048,.068,-.068)
                       for dx in (.023,-.023,.050,-.050)]
            def placement_cost(offset):
                pt = base + offset
                bounds = 50 if not (.025 < pt[0] < .975 and .025 < pt[1] < .975) else 0
                occupied = sum(np.linalg.norm((pt-other)*[1.0,1.4]) < .041 for other in chosen)
                crowd = sum(np.linalg.norm((pt-pos[node])*[1.,1.4]) < .024 for node in graph if node != gene)
                return bounds + 8*occupied + crowd + 10*np.linalg.norm(offset)
            offset = min(offsets,key=placement_cost)
            dest=base+offset
            ax.annotate(gene,base,xytext=dest,ha="center",va="center",fontsize=7.4,
                fontweight="bold" if completed else "normal",color="#245771" if completed else "#273740",
                bbox=dict(facecolor="white",edgecolor="none",pad=.5,alpha=.88),
                arrowprops=dict(arrowstyle="-",linewidth=.5,color="#77868b",shrinkA=1,shrinkB=2))
            chosen.append(dest)
        title="Prior-supported connections between strict-core components" if completed else "Basal-like strict core across five training runs"
        fig.suptitle(title,fontsize=18,y=.98,fontweight="semibold")
        handles=[Line2D([],[],marker="o",linestyle="",color=c,label=f"{label} ({len(n)})")
                 for c,label,n in zip(COLORS,CATEGORIES,cats)]
        handles.extend(Line2D([],[],color=c,lw=1.5,label=f"{label} ({int(sum(edges.edge_type==i))})")
            for i,(c,label) in enumerate(zip(EDGE_COLORS,["HuRI","OmniPath activation","OmniPath inhibition"])))
        if completed:
            handles.extend([Line2D([],[],color="#3678aa",linestyle="--",label="Added prior edges (75)"),
                            Line2D([],[],color="#3678aa",marker="D",linestyle="",label="Added connector genes (4)")])
        fig.legend(handles=handles,loc="lower center",ncol=3,fontsize=9,frameon=False,bbox_to_anchor=(.5,.012))
        for ext in ("pdf","svg","png"):
            fig.savefig(OUT/f"FigureS{5 if completed else 4}.{ext}",dpi=180)
        plt.close(fig)

if __name__=="__main__":
    main()
