import os, io, base64
from h2o_wave import main, app, Q, ui, on, handle_on, data
from typing import Optional, List
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from tabulate import tabulate

FIGSIZE=(6, 2)
GRID_IMG_SIZE = '400px'

# Use for page cards that should be removed when navigating away.
# For pages that should be always present on screen use q.page[key] = ...
def add_card(q, name, card) -> None:
    q.client.cards.add(name)
    q.page[name] = card


# Remove all the cards related to navigation.
def clear_cards(q, ignore: Optional[List[str]] = []) -> None:
    if not q.client.cards:
        return
    for name in q.client.cards.copy():
        if name not in ignore:
            del q.page[name]
            q.client.cards.remove(name)


@on('#page1')
async def page1(q: Q):

    # Move to train page if the user clicks Next button after uploading data
    if q.args.move_to_train:
        q.page['header']['tabs'].value='#page2'
        await page2(q)
        return


    links = q.args.user_files

    clear_cards(q)  # When routing, drop all the cards except of the main ones (header, sidebar, meta).
    add_card(q, 'info', ui.tall_info_card(box='vertical', name='', title='Import Data',
                                                  caption='Import your data files', icon='LandscapeOrientation'))

    if links:
        items = []
        for link in links:
            local_path = await q.site.download(link, f'./datasets/{os.path.basename(link)}')
            #
            # The file is now available locally; process the file.
            # To keep this example simple, we just read the file size.
            #
            size = os.path.getsize(local_path)

            # items.append(ui.link(label=f'{os.path.basename(link)} ({size} bytes)', download=True, path=link))
            items.append(ui.text_xl(f'File uploaded: {os.path.basename(link)} ({size} bytes)'))
            
            # Clean up
            # os.remove(local_path)

        add_card(q, 'file_upload', ui.form_card(box='vertical', items=[
                *items,
                ui.button(name='move_to_train', label='Next', primary=True)
            ]))

    else:
        add_card(q, 'file_upload', ui.form_card(box='vertical', items=[
            ui.file_upload(name='user_files', label='Upload', multiple=True),
        ]))

    
    
    


@on('#page2')
async def page2(q: Q):
    clear_cards(q)  # When routing, drop all the cards except of the main ones (header, sidebar, meta).

    # Move to train page if the user clicks Next button after uploading data
    if q.args.run_model:
        await start_training(q)
        return
    
    if q.args.show_results:
        q.page['header']['tabs'].value='#page3'
        await page3(q)
        return

    # Read all available .csv file names
    data_files = os.listdir('./datasets/')
    q.user.data_file = data_file = q.user.data_file or data_files[0] if len(data_files) > 0 else None
    
    if q.args.data_file:
        q.user.data_file = data_file = q.args['data_file']

    q.user.train_split = train_split = q.user.train_split or 0.8

    if q.args.train_split:
        q.user.train_split = train_split = q.args['train_split']

    rows = []

    # Get the column
    if data_file:
        df = pd.read_csv(f'./datasets/{data_file}')
        rows = list(df.columns.values)

    if q.user.target_col in rows:
        q.user.target_col = target_col = q.args['target_col']
    else:
        q.user.target_col = target_col = rows[0] if len(rows) > 0 else None


    max_models = q.user.max_models or 10

    if q.args.max_models:
        q.user.max_models = max_models = q.args['max_models']

    max_run_time = q.user.max_run_time or 0

    if q.args.max_run_time is not None:
        q.user.max_run_time = max_run_time = q.args['max_run_time']

    
    stop_metric = q.user.stop_metric or 'AUTO'

    if q.args.stop_metric:
        q.user.stop_metric = stop_metric = q.args['stop_metric']

    
    algos = q.user.algos or []
    q.user.algos = algos = q.args['algos']
        


    
    

    add_card(q, 'train_settings', ui.form_card(box='vertical', items=[
            ui.dropdown(name='data_file', label='Dataset', required=True, value=data_file, trigger=True, choices=[
                ui.choice(name=data_file, label=data_file) for data_file in data_files
            ]),
            ui.spinbox(name='train_split', label='Train Split',min=0.2, max=0.9, step=0.1, value=train_split, trigger=True),
            ui.dropdown(name='target_col', label='Target Column', required=True, value=target_col, trigger=True, choices=[
                ui.choice(name=row, label=row) for row in rows
            ]),
            ui.spinbox(name='max_models', label='Max Models',min=0, max=20, step=1, value=max_models, trigger=True),
            ui.slider(name='max_run_time', label='Maxium Runtime (s)', value=max_run_time, min=0, max=3600, trigger=True),
            ui.dropdown(name='stop_metric', label='Stop Metric', value=stop_metric, trigger=True, choices=[
                ui.choice(name='AUTO', label='Auto'),
                ui.choice(name='deviance', label='Deviance'),
                ui.choice(name='MSE', label='Mean Squared Error'),
                ui.choice(name='MAE', label='Mean Absolute Error'),
                ui.choice(name='AUC', label='Area Under the ROC Curve'),
            ]),
            ui.picker(name='algos', label='Algorithms', trigger=True, choices=[
                ui.choice(name='DRF', label='Distributed Random Forest (DRF)'),
                ui.choice(name='GLM', label='Generalized Linear Model (GLM)'),
                ui.choice(name='GBM', label='Gradient Boosting Machine (GBM)'),
                ui.choice(name='DeepLearning', label='Deep Learning'),
            ], values=algos),
            ui.button(name='run_model', label='Run Model', primary=True)
        ]))

    


@on('#page3')
async def page3(q: Q):
    clear_cards(q)  

    q.user.pf_plot  = q.user.aml.pareto_front(test_frame=q.app.validation_data, figsize=(FIGSIZE[0], FIGSIZE[0]))
    q.user.mc_plot = q.user.aml.model_correlation_heatmap(frame=q.app.validation_data, figsize=(FIGSIZE[0], FIGSIZE[0]))
    q.user.vh_plot = q.user.aml.varimp_heatmap(figsize=(FIGSIZE[0], FIGSIZE[0]))
    
    # Display leadboard
    lb = q.user.aml.leaderboard
    colums = lb.columns
    
    
    add_card(q, 'detail', ui.form_card(box='vertical', items=[
        ui.text_l('Model Leader Board'),
        ui.table(
            name='table',
            columns=[
                ui.table_column(name=column, label=column) for column in colums
            ], 
            rows=[
                ui.table_row(name=f'row{i}', cells=[lb[i, 0], *[str(x) for x in lb[i, 1:].getrow()]]) for i in range(1, lb.nrows)
            ]
        )
    ]))


    # Plot leader's stats
    q.user.shap_sum = q.user.aml.leader.shap_summary_plot(frame=q.app.validation_data, figsize=(FIGSIZE[0], FIGSIZE[0]))
    q.user.learning_curve = q.user.aml.leader.learning_curve_plot(figsize=(FIGSIZE[0], FIGSIZE[0]))


    add_card(q, 'img_pf_plot', ui.image_card(box=ui.box('grid', width=GRID_IMG_SIZE),title="Pareto Front Plot", type="png",
                image=get_image_from_matplotlib(q.user.pf_plot)
                ))
    
    add_card(q, 'img_mc_plot', ui.image_card(box=ui.box('grid', width=GRID_IMG_SIZE),title="Model correlation heatmap", type="png",
                image=get_image_from_matplotlib(q.user.mc_plot)
                ))
    
    add_card(q, 'img_vh_plot', ui.image_card(box=ui.box('grid', width=GRID_IMG_SIZE),title="Varimp Heatmap", type="png",
                image=get_image_from_matplotlib(q.user.vh_plot)
                ))
    
    add_card(q, 'img_shap_sum', ui.image_card(box=ui.box('grid', width=GRID_IMG_SIZE),title="SHAP Summary", type="png",
                image=get_image_from_matplotlib(q.user.shap_sum)
                ))
    
    add_card(q, 'img_learning_curve', ui.image_card(box=ui.box('grid', width=GRID_IMG_SIZE),title="Leraning Curve", type="png",
                image=get_image_from_matplotlib(q.user.learning_curve)
                ))
    

@on('#page4')
async def handle_page4(q: Q):
    # When routing, drop all the cards except of the main ones (header, sidebar, meta).
    # Since this page is interactive, we want to update its card instead of recreating it every time, so ignore 'form' card on drop.
    clear_cards(q, ['form'])

    if q.args.step1:
        # Just update the existing card, do not recreate.
        q.page['form'].items = [
            ui.stepper(name='stepper', items=[
                ui.step(label='Step 1'),
                ui.step(label='Step 2'),
                ui.step(label='Step 3'),
            ]),
            ui.textbox(name='textbox2', label='Textbox 1'),
            ui.buttons(justify='end', items=[
                ui.button(name='step2', label='Next', primary=True),
            ])
        ]
    elif q.args.step2:
        # Just update the existing card, do not recreate.
        q.page['form'].items = [
            ui.stepper(name='stepper', items=[
                ui.step(label='Step 1', done=True),
                ui.step(label='Step 2'),
                ui.step(label='Step 3'),
            ]),
            ui.textbox(name='textbox2', label='Textbox 2'),
            ui.buttons(justify='end', items=[
                ui.button(name='step1', label='Cancel'),
                ui.button(name='step3', label='Next', primary=True),
            ])
        ]
    elif q.args.step3:
        # Just update the existing card, do not recreate.
        q.page['form'].items = [
            ui.stepper(name='stepper', items=[
                ui.step(label='Step 1', done=True),
                ui.step(label='Step 2', done=True),
                ui.step(label='Step 3'),
            ]),
            ui.textbox(name='textbox3', label='Textbox 3'),
            ui.buttons(justify='end', items=[
                ui.button(name='step2', label='Cancel'),
                ui.button(name='submit', label='Next', primary=True),
            ])
        ]
    else:
        # If first time on this page, create the card.
        add_card(q, 'form', ui.form_card(box='vertical', items=[
            ui.stepper(name='stepper', items=[
                ui.step(label='Step 1'),
                ui.step(label='Step 2'),
                ui.step(label='Step 3'),
            ]),
            ui.textbox(name='textbox1', label='Textbox 1'),
            ui.buttons(justify='end', items=[
                ui.button(name='step2', label='Next', primary=True),
            ]),
        ]))


async def init(q: Q) -> None:
    q.page['meta'] = ui.meta_card(box='', layouts=[ui.layout(breakpoint='xs', min_height='100vh', zones=[
        ui.zone('header'),
        ui.zone('content', zones=[
            # Specify various zones and use the one that is currently needed. Empty zones are ignored.
            ui.zone('horizontal', direction=ui.ZoneDirection.ROW),
            ui.zone('vertical'),
            ui.zone('grid', direction=ui.ZoneDirection.ROW, wrap='stretch', justify='center')
        ]),
    ])])
    q.page['header'] = ui.header_card(
        box='header', title='My app', subtitle="Let's conquer the world",
        image='https://wave.h2o.ai/img/h2o-logo.svg',
        secondary_items=[
            ui.tabs(name='tabs', value=f'#{q.args["#"]}' if q.args['#'] else '#page1', link=True, items=[
                ui.tab(name='#page1', label='Import'),
                ui.tab(name='#page2', label='Train'),
                ui.tab(name='#page3', label='Results'),
                ui.tab(name='#page4', label='Form'),
            ]),
        ],

    
        items=[
            ui.persona(title='Neethamadhu Madurasinghe', subtitle='Developer', size='xs',
                       image='https://media.licdn.com/dms/image/D5603AQG4N4-azNSgYQ/profile-displayphoto-shrink_200_200/0/1690195082256?e=1696464000&v=beta&t=vvKyw7lDG6eA3EdC0sqxRim3Kki3zIbwRNcj47cnhSo'),
        ]
    )
    # If no active hash present, render page1.
    if q.args['#'] is None:
        await page1(q)


@app('/')
async def serve(q: Q):
    # Run only once per client connection.
    if not q.client.initialized:
        q.client.cards = set()
        await init(q)
        q.client.initialized = True

    # Handle routing.
    await handle_on(q)
    await q.page.save()


        
async def start_training(q: Q):
    h2o.init()
    df = pd.read_csv(f'./datasets/{q.user.data_file}')
    df = h2o.H2OFrame(df)

    train, validation = df.split_frame(ratios=[q.user.train_split])
    q.app.validation_data = validation

    x = train.columns
    y = q.user.target_col
    if y in x:
        x.remove(y)


    max_models = q.user.max_models
    max_run_time = q.user.max_run_time
    if max_run_time == 0:
        max_run_time = None

    stop_metric = q.user.stop_metric
    algos = q.user.algos

    if algos == []:
        algos = None

    add_card(q, 'info', ui.tall_info_card(box='vertical', name='prog', title='Training',
                                                  caption='Please wait...', icon='ProcessingRun'))

    add_card(q, 'progress', ui.form_card(box='vertical',
        items=[ui.progress(label='Training', caption='Please wait...')]
    ))

    train_details = f'''
        Source: {q.user.data_file}
        Trainning Split: {q.user.train_split}
        Target: {y}
        Max Runtime: {max_run_time if max_run_time else 'UNLIMITED'}
        Max Models: {max_models}
        Stop Metric: {stop_metric}
        Algorithms: {algos if algos else 'AUTO'}

     '''

    add_card(q, 'detail', ui.markdown_card(
    box='vertical',
    title='Parameters',
    content=train_details
    ))

    await q.page.save()
    q.user.aml = H2OAutoML(max_models=max_models, seed=1, max_runtime_secs=max_run_time, stopping_metric=stop_metric, include_algos=algos)
    q.user.aml.train(x=x, y=y, training_frame=train, validation_frame=validation)


    
    clear_cards(q)
    add_card(q, 'info', ui.tall_info_card(box='vertical', name='', title='Completed Training',
                                                  caption='Click next to see results', icon='StatusCircleCheckmark'))
    

    add_card(q, 'train_settings', ui.form_card(box='vertical', items=[
            ui.button(name='show_results', label='Next', primary=True)
        ]))


    return


    # Image from matplotlib object
def get_image_from_matplotlib(matplotlib_obj):
    if hasattr(matplotlib_obj, "figure"):
        matplotlib_obj = matplotlib_obj.figure()
    buffer = io.BytesIO()
    matplotlib_obj.savefig(buffer, format="png")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

