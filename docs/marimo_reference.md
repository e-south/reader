## Marimo notebooks (context for agents)

This document is a reference for working with marimo notebooks.

Where this fits:
- For the overall workflow, see [README](../README.md).
- For notebook usage, see [docs/notebooks.md](./notebooks.md).
- For pipeline outputs and plots/exports that notebooks consume, see [docs/pipeline.md](./pipeline.md).
- Use `reader notebook` to scaffold an experiment notebook with paths and manifests wired up.

Editing rule: only edit code inside the `@app.cell` function body. Marimo manages parameters and return statements.

```python
@app.cell
def _():
    # edit HERE
    ...
    return
```

---

### Marimo fundamentals

Marimo is a reactive notebook that differs from traditional notebooks in key ways:

- Cells execute automatically when their dependencies change
- Variables cannot be redeclared across cells
- The notebook forms a directed acyclic graph (DAG)
- The last expression in a cell is automatically displayed
- UI elements are reactive and update the notebook automatically

---

### Code Requirements

1. All code must be complete and runnable
2. Follow consistent coding style throughout
3. Include descriptive variable names and helpful comments
4. Import all modules in the first cell, always including `import marimo as mo`
5. Never redeclare variables across cells
6. Ensure no cycles in notebook dependency graph
7. The last expression in a cell is automatically displayed, just like in Jupyter notebooks.
8. Don't include comments in markdown cells
9. Don't include comments in SQL cells
10. Never define anything using `global`.

---

### Reactivity

Marimo's reactivity means:

- When a variable changes, all cells that use that variable automatically re-execute
- UI elements trigger updates when their values change without explicit callbacks
- UI element values are accessed through `.value` attribute
- You cannot access a UI element's value in the same cell where it's defined
- Cells prefixed with an underscore (e.g. _my_var) are local to the cell and cannot be accessed by other cells

---

### Data handling

* Prefer `polars` for data manipulation.
* Validate inputs early (schema, required columns).
* Handle missing values intentionally.
* Keep transformations efficient.
* If a `polars.DataFrame` is the last expression in a cell, it displays nicely.

---

### Visualization

* Matplotlib: make `plt.gca()` the last expression instead of calling `plt.show()`.
* Plotly: return the figure object directly.
* Altair: return the chart object directly; add tooltips when appropriate.
  * You can pass polars DataFrames directly to Altair in most cases.
* Always include labels, titles, and readable scales.

---

### UI elements

* Access values with `.value` (e.g., `slider.value`).
* Define UI elements in one cell and reference them later (don’t read `.value` in the same cell).
* Use `mo.hstack`, `mo.vstack`, `mo.tabs` for layout.
* Prefer reactive updates over callbacks (marimo handles reactivity automatically).
* Group related controls to improve usability.

---

### SQL

* For DuckDB-style queries, prefer marimo SQL cells.
* Convention:

  ```python
  df = mo.sql(f"""
  SELECT ...
  """)
  ```
* Don’t add comments in cells that use `mo.sql()`.

---

### Troubleshooting

Common issues and solutions:

* Circular dependencies:

  * Reorganize cells to remove cycles.
  * Split “UI definition” and “UI consumption” into separate cells.
* UI element value access:

  * Move `.value` reads into a separate dependent cell.
* Visualization not showing:

  * Ensure the visualization object (axes/figure/chart) is the last expression in the cell.

After generating a notebook, run:

```bash
marimo check --fix
```

to catch common formatting issues and pitfalls (when available in your environment).

---

### Available UI elements

- `mo.ui.altair_chart(altair_chart)`
- `mo.ui.button(value=None, kind='primary')`
- `mo.ui.run_button(label=None, tooltip=None, kind='primary')`
- `mo.ui.checkbox(label='', value=False)`
- `mo.ui.date(value=None, label=None, full_width=False)`
- `mo.ui.dropdown(options, value=None, label=None, full_width=False)`
- `mo.ui.file(label='', multiple=False, full_width=False)`
- `mo.ui.number(value=None, label=None, full_width=False)`
- `mo.ui.radio(options, value=None, label=None, full_width=False)`
- `mo.ui.refresh(options: List[str], default_interval: str)`
- `mo.ui.slider(start, stop, value=None, label=None, full_width=False, step=None)`
- `mo.ui.range_slider(start, stop, value=None, label=None, full_width=False, step=None)`
- `mo.ui.table(data, columns=None, on_select=None, sortable=True, filterable=True)`
- `mo.ui.text(value='', label=None, full_width=False)`
- `mo.ui.text_area(value='', label=None, full_width=False)`
- `mo.ui.data_explorer(df)`
- `mo.ui.dataframe(df)`
- `mo.ui.plotly(plotly_figure)`
- `mo.ui.tabs(elements: dict[str, mo.ui.Element])`
- `mo.ui.array(elements: list[mo.ui.Element])`
- `mo.ui.form(element: mo.ui.Element, label='', bordered=True)`

### Layout and utility functions

- `mo.md(text)` - display markdown
- `mo.stop(predicate, output=None)` - stop execution conditionally
- `mo.output.append(value)` - append to the output when it is not the last expression
- `mo.output.replace(value)` - replace the output when it is not the last expression
- `mo.Html(html)` - display HTML
- `mo.image(image)` - display an image
- `mo.hstack(elements)` - stack elements horizontally
- `mo.vstack(elements)` - stack elements vertically
- `mo.tabs(elements)` - create a tabbed interface

---

### Example: Markdown cell

```python
@app.cell
def _():
    import marimo as mo
    mo.md("""
# Hello world
This is a _markdown cell_.
""")
    return
```

---

### Example: Basic UI with reactivity

```python
@app.cell
def _():
    import marimo as mo
    import altair as alt
    import polars as pl
    import numpy as np
    return mo, alt, pl, np
```

```python
@app.cell
def _(mo):
    n_points = mo.ui.slider(10, 100, value=50, label="Number of points")
    n_points
    return n_points
```

```python
@app.cell
def _(alt, pl, np, n_points):
    x = np.random.rand(n_points.value)
    y = np.random.rand(n_points.value)
    df = pl.DataFrame({"x": x, "y": y})

    chart = (
        alt.Chart(df)
        .mark_circle(opacity=0.7)
        .encode(
            x=alt.X("x", title="X axis"),
            y=alt.Y("y", title="Y axis"),
        )
        .properties(
            title=f"Scatter plot with {n_points.value} points",
            width=400,
            height=300,
        )
    )
    chart
    return df, chart
```

---

### Example: Data explorer

```python
@app.cell
def _():
    import marimo as mo
    import polars as pl
    from vega_datasets import data
    return mo, pl, data
```

```python
@app.cell
def _(mo, pl, data):
    cars_df = pl.DataFrame(data.cars())
    mo.ui.data_explorer(cars_df)
    return cars_df
```

---

### Example: Multiple UI elements

```python
@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    return mo, pl, alt
```

```python
@app.cell
def _(pl):
    iris = pl.read_csv("hf://datasets/scikit-learn/iris/Iris.csv")
    return iris
```

```python
@app.cell
def _(mo, iris):
    species_selector = mo.ui.dropdown(
        options=["All"] + iris["Species"].unique().to_list(),
        value="All",
        label="Species",
    )
    x_feature = mo.ui.dropdown(
        options=iris.select(pl.col(pl.Float64, pl.Int64)).columns,
        value="SepalLengthCm",
        label="X Feature",
    )
    y_feature = mo.ui.dropdown(
        options=iris.select(pl.col(pl.Float64, pl.Int64)).columns,
        value="SepalWidthCm",
        label="Y Feature",
    )
    mo.hstack([species_selector, x_feature, y_feature])
    return species_selector, x_feature, y_feature
```

```python
@app.cell
def _(alt, iris, species_selector, x_feature, y_feature):
    import polars as pl

    filtered = (
        iris
        if species_selector.value == "All"
        else iris.filter(pl.col("Species") == species_selector.value)
    )

    chart = (
        alt.Chart(filtered)
        .mark_circle()
        .encode(
            x=alt.X(x_feature.value, title=x_feature.value),
            y=alt.Y(y_feature.value, title=y_feature.value),
            color="Species",
            tooltip=["Species", x_feature.value, y_feature.value],
        )
        .properties(
            title=f"{y_feature.value} vs {x_feature.value}",
            width=500,
            height=400,
        )
    )
    chart
    return filtered, chart
```

---

### Example: Conditional outputs

```python
@app.cell
def _(mo, data, mode):
    mo.stop(not data.value, mo.md("No data to display"))

    if mode.value == "scatter":
        mo.output.replace(render_scatter(data.value))
    else:
        mo.output.replace(render_bar_chart(data.value))
    return
```

---

### Example: Interactive chart with Altair

```python
@app.cell
def _():
    import marimo as mo
    import altair as alt
    import polars as pl
    return mo, alt, pl
```

```python
@app.cell
def _(alt, pl):
    weather = pl.read_csv(
        "https://raw.githubusercontent.com/vega/vega-datasets/refs/heads/main/data/weather.csv"
    )
    weather_dates = weather.with_columns(
        pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d")
    )

    base_chart = (
        alt.Chart(weather_dates)
        .mark_point()
        .encode(
            x="date:T",
            y="temp_max",
            color="location",
            tooltip=["location", "temp_max", "date"],
        )
    )
    return weather_dates, base_chart
```

```python
@app.cell
def _(mo, base_chart):
    chart = mo.ui.altair_chart(base_chart)
    chart
    return chart
```

```python
@app.cell
def _(chart):
    # Use chart.value as needed
    chart.value
    return
```

---

### Example: Run buttons

```python
@app.cell
def _(mo):
    first_button = mo.ui.run_button(label="Option 1")
    second_button = mo.ui.run_button(label="Option 2")
    [first_button, second_button]
    return first_button, second_button
```

```python
@app.cell
def _(first_button, second_button):
    if first_button.value:
        print("You chose option 1!")
    elif second_button.value:
        print("You chose option 2!")
    else:
        print("Click a button!")
    return
```

---

### Example: SQL with DuckDB (via marimo SQL)

```python
@app.cell
def _():
    import marimo as mo
    import polars as pl
    return mo, pl
```

```python
@app.cell
def _(pl):
    weather = pl.read_csv(
        "https://raw.githubusercontent.com/vega/vega-datasets/refs/heads/main/data/weather.csv"
    )
    return weather
```

```python
@app.cell
def _(mo, weather):
    seattle_weather_df = mo.sql(
        f"""
        SELECT *
        FROM weather
        WHERE location = 'Seattle';
        """
    )
    return seattle_weather_df
```

---

@e-south
