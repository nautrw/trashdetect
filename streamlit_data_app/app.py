from collections import Counter
import streamlit as st
import pandas as pd
from ultralytics.data.utils import check_det_dataset
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

dataset_dir = Path.cwd().parent.parent / "datasets" / "taco"
data = check_det_dataset(dataset_dir / "data.yaml")
names = data["names"]

@st.cache_data(show_time=False, show_spinner=False)
def count_classes(data: dict) -> dict[str, Counter]:
    splits = ["train", "val", "test"]
    result = {}

    with st.status("Counting labels..."):
        for split in splits:
            if split not in data:
                continue

            st.write(f"Counting `{split}` section labels...")
            progress_bar = st.progress(0)
            
            # img_dir = Path(data[split]) # it defaults to /images
            label_dir = Path(data[split]).parent / "labels"
            
            counter = Counter() # special dict, like defaultdict for int
            label_files = list(label_dir.glob("*.txt"))
            for i, label in enumerate(label_files):
                progress = (i+1) / len(label_files)
                progress_bar.progress(progress, text=f"Reading file: {label} ({i+1}/{len(label_files)})")
                
                for line in label.read_text().splitlines():
                    if line.strip():
                        class_id = int(line.split()[0])
                        counter[class_id] += 1
                        
            result[split] = counter
        
        st.write("Done!")

        return result

classes_count = count_classes(data)

splits = []
values = []
classes = []

for split in classes_count:
    counter = classes_count[split]
    
    for class_id in counter:
        splits.append(split)
        values.append(counter[class_id])
        classes.append(list(names.values())[class_id])

df = pd.DataFrame({"Class": classes, "Total": values, "Split": splits})

class_order = (
    df.groupby("Class")["Total"]
    .sum()
    .sort_values(ascending=False)
    .index
    .tolist()    
)

fig = px.bar(df, title="Class Samples Per Split", y="Split", x="Total", color="Class", orientation='h', category_orders={"Class": class_order})
fig.update_xaxes(categoryorder="total descending")

totals = df.groupby("Split")["Total"].sum().reset_index()

# Add totals at the end of each stacked bar
fig.add_trace(
    go.Scatter(
        x=totals["Total"],
        y=totals["Split"],
        text=totals["Total"],
        mode="text",
        textposition="middle right",
        showlegend=False,
        hoverinfo="skip",
    )
)

st.plotly_chart(fig)

overall_split_tab, train_split_tab, val_split_tab, test_split_tab = st.tabs(["Total", "Train", "Validation", "Testing"])
with overall_split_tab:
    fig = px.pie(df, names=df.Class, values=df.Total, title="Overall Categories")
    st.plotly_chart(fig)
with train_split_tab:
    train_split_df = df.where(df.Split == "train").dropna()
    fig = px.pie(train_split_df, names=train_split_df.Class, values=train_split_df.Total, title="Train Split Categories")
    st.plotly_chart(fig)
with val_split_tab:
    val_split_df = df.where(df.Split == "val").dropna()
    fig = px.pie(val_split_df, names=val_split_df.Class, values=val_split_df.Total, title="Validation Split Categories")
    st.plotly_chart(fig)
with test_split_tab:
    test_split_df = df.where(df.Split == "test").dropna()
    fig = px.pie(test_split_df, names=test_split_df.Class, values=test_split_df.Total, title="Test Split Categories")
    st.plotly_chart(fig)

# df = pd.DataFrame({"Classes": classes, "Values": values})
# fig = px.pie(df, title="Classes and Values", names=classes, values=values)
# st.plotly_chart(fig)