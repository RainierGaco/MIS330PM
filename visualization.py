TypeError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/project-management/visualization.py", line 181, in <module>
    st.plotly_chart(fig_stack, use_container_width=True)
File "/home/adminuser/venv/lib/python3.12/site-packages/streamlit/runtime/metrics_util.py", line 444, in wrapped_func
    result = non_optional_func(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/streamlit/elements/plotly_chart.py", line 501, in plotly_chart
    plotly_chart_proto.spec = plotly.io.to_json(figure, validate=False)
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/plotly/io/_json.py", line 221, in to_json
    return to_json_plotly(fig_dict, pretty=pretty, engine=engine)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/plotly/io/_json.py", line 142, in to_json_plotly
    json.dumps(plotly_object, cls=PlotlyJSONEncoder, **opts), _swap_json
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/json/__init__.py", line 238, in dumps
    **kw).encode(obj)
          ^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/_plotly_utils/utils.py", line 162, in encode
    encoded_o = super(PlotlyJSONEncoder, self).encode(o)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/json/encoder.py", line 200, in encode
    chunks = self.iterencode(o, _one_shot=True)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/json/encoder.py", line 258, in iterencode
    return _iterencode(o, 0)
           ^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/_plotly_utils/utils.py", line 239, in default
    return _json.JSONEncoder.default(self, obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/json/encoder.py", line 180, in default
    raise TypeError(f'Object of type {o.__class__.__name__} 
