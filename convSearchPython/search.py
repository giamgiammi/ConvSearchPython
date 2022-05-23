"""
This is the main executable for the project.
It provides the command line interface to run pipelines and save the results.

# Command line arguments

This script can be called alternatively using:

- `python -m convSearchPython.search`

    While your current folder is the main folder of the project (ConvSearchPython)
    or you have added it to PYTHONPATH.

- `python /path/to/ConvSearchPython/convSearchPython/search.py`

    As the script will automatically add its main folder to the python path
    upon execution

The calling syntax is:

    usage: search.py [-h] [-p, --processes PROCS] PROPS

    positional arguments:
      PROPS                 Search properties (ini file)

    optional arguments:
      -h, --help            show this help message and exit
      -p, --processes PROCS
                            Number of worker processes (default to 1)


# Search properties file

The properties file that configure the search is in INI format and
is composed by 3 section: 'SETTINGS', 'PIPELINES' and 'FACTORIES'.
The first one is always mandatory, while at least one of the other two must exists.

Below follows the description of each section.

## SETTINGS

* `dataset.fn` (string, **mandatory**)

    Provider for the topic dataset.

    Full reference of a function that will provide a tuple composed of

    1. `queries: DataFrame`
    2. `qrels: DataFrame`
    3. `conversations: Dict[str, List[str]]`
    4. `query_map: Dict[str, Tuple[str, int]]`

    `conversations` must be a dict that map the conversation id to a list
    of qid for the query inside that conversation.

    `query_map` must be a sict that map a query qid to a tuple where the first
    value is the conversation id and the second the index of this conversation
    in the conversation list.

* `dataset.value` (string, optional)

    If provided is the value passed to `dataset.fn`

* `include.text` (boolean, optional)

    Whether include document text in the run. Default to false.
* `index` (string, **mandatory**)

    Name of the index as defined inside config.ini.

* `run.name` (string, optional)

    Name for the run. Default to 'test'.

* `run.trec` (boolean, optional)

    Whether save the run in trec_eval format. Default to true.

* `run.csv` (boolean, optional)

    Whether save the run in csv format. Default to false.

* `run.pkl` (boolean, optional)

    Whether save the run in pkl (python pickle) format. Default to false.

* `run.limit` (integer, optional)

    Limit the number of results per qid.
    Negative (or 0) numbers disable this function.
    Default to -1.

* `measure.txt` (boolean, optional)

    Whether save the results' measures in a human-readable format. Default to true.

* `measure.csv` (boolean, optional)

    Whether save the results' measures in csv format. Default to false.

* `measure.pkl` (boolean, optional)

    Whether save the results' measures in pkl (python pickle) format. Default to false.

* `measure.parsable` (boolean, optional)

    Whether save the results' measure in an easy-to-parse csv format divided by measure.
    Default to true.

* `metrics` (string, optional)

    Comma-separated list of measure to calculate
    Refer to pyterrier documentation [Available Evaluation Measures](https://pyterrier.readthedocs.io/en/latest/experiments.html#available-evaluation-measures)
    for the possible values.

    Default to 'AP, nDCG@3, P@1, P@3, RR, R@200, R@100'.

* `pipelines` (string, **mandatory**)

    Comma-separated list of pipelines to run. The name can refer to a pipeline or a pipeline factory.
    Both possibility are discussed in the next two sections.

### Example

<details>
  <summary>expand/collapse</summary>
```
[SETTINGS]

; Full reference of function to load the dataset
dataset.fn = convSearchPython.dataset.CAsT.cast_dataset

; Value to pass to the dataset function
dataset.value = 2019

; Include text inside runs (true/false)
include.text = false

; Index as named in config.ini
index = myIndex

; Name of the run
run.name = test

; Save run as trec/csv/pkl (true/false)
run.trec = true
run.csv = false
run.pkl = false

; limit run results per qid (-1 = disabled)
run.limit = -1

; Save measures as txt/csv/pkl (true/false)
measures.txt = true
measures.csv = false
measures.pkl = false

; Easy to parse measures divided by metric
measures.parsable = true

; Metrics to consider (comma-separated)
metrics = AP, nDCG@3, P@1, P@3, RR, R@200, R@100

; Pipelines to run (comma-separated)
pipelines = pipe1, pipe2, pipe3
```
</details>

## PIPELINES

This section is used to initialize Pipeline object.

A Pipeline object is a subclass of `convSearchPython.pipelines.Pipeline`
that allow its writer to provide a custom logic to combine different
models, rewriters and rerankers.

## Keys

The keys in this section are composed by a sequence of names concatenated with a dot.
The symbol '$' is reserved, so you shouldn't use it for custom names.

- The first name is always the pipeline assigned name. The key with only the first name must
have the full class reference as (single) value.

- The second name is the name of an argument that should be passed to the constructor.

- The third name, if present, can be '$range' or '$file'. These two special
options tell the parser that the value is, respectively, a range or a separated params file.

## Values

There are four different modalities to provide values:

- Single values

    Just write the value after the '=' symbol. You **must not** use commas (,)
    in the value unless you put everything between quotes (")

- List of values

    Write the values in a comma-separated list.
    If you need to use the comma inside a value, put the value in quotes.

- Range of values

    Append .$range to the key.
    Write the value as `start, stop, step`.
    Stop is included (if reached) and float value are supported.

- Param file

    Append .$file to the key.
    Use the path of the params file as value. If the path is relative,
    will be considered relative to the PROPS file folder.

    The params file is itself in INI format. It might contain two sections:
    'RANGE' and 'VALUE'. Both use the argument name as key.
    Inside RANGE the value must be in form `start, stop, step`.
    Inside VALUE the value must be a comma-separated list of values.

All the arguments provided by these methods as merged together.
Arguments will multiple values will result in multiple pipelines with every combination
of value possible.

### Example

<details>
  <summary>expand/collapse</summary>
```
[PIPELINES]

pipe3 = full.reference.to.class

; single value
pipe3.p1 = 5

; list of values
pipe3.p2 = 5, 10, 15

; range of values [3, 5, 7, 9, 11]
pipe3.p3.$range =  3, 11, 2

; force string value
pipe3.p4 = "100"

; param file
pipe3.$file = params.ini
```
</details>

<details>
  <summary>params file</summary>
```
[RANGE]

;param = start, stop, step (stop included if reached)
fb_lambda = 0.4, 0.7, 0.1

[VALUE]

;option = value1, value2, value3
fb_docs = 10, 15, 20
fb_lambda = 0.9
rm3 = true, false
```
</details>

## FACTORIES

This section allows constructing pipelines using factories.

The syntax is the same, but the referenced class must be a Pipeline Factory.
Other options are factory-dependent.

For the provided factory classes, check `convSearchPython.pipelines.factory`.

### Examples

<details>
<summary>Expand/Collapse</summary>
```
[FACTORIES]

pipe1 = convSearchPython.pipelines.factory.base_factory.BasePipelineFactory
pipe1.model = DirichletLM
pipe1.model.mu = 2500
pipe1.rm3.terms = 10
pipe1.rm3.docs = 10
pipe1.rm3.lambda = 0.5
pipe1.rewriter = full.path.to.rewriter.class
pipe1.rewriter.p1 = 3
pipe1.rewriter.p2 = 3, 5, 10
pipe1.rewriter.p3.$range = 2, 10, 2
pipe1.reranker = full.path.to.reranker.class
pipe1.reranker.$file = params.ini

pipe2 = convSearchPython.pipelines.SubIndexPipelineFactory
pipe2.base-model = BM25
pipe2.base-model.c = 0.7
pipe2.rewriter = full.path...
pipe2.reranker = full.path...
```
</details>
"""


def _proc_init():
    import multiprocessing as mp
    from os import environ
    from sys import stderr, stdout

    print(f'A new worker process started: {mp.current_process().name}', flush=True)
    environ['WORKER_PROCESS'] = 'true'
    # noinspection PyUnresolvedReferences
    import convSearchPython.basics
    stdout.flush()
    stderr.flush()


if __name__ == '__main__':
    import itertools
    import multiprocessing as mp
    import sys
    from argparse import ArgumentParser
    from collections import defaultdict
    from datetime import timedelta
    from pathlib import Path
    from time import time
    from typing import NamedTuple, Callable, Tuple, List, Any, Dict, Union, Mapping, Iterator

    from pandas import DataFrame

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from convSearchPython.utils.ConfigParserWrapper import ConfigParser, SectionProxy
    from convSearchPython.pipelines import Pipeline
    from convSearchPython.pipelines.factory import PipelineFactory
    from convSearchPython.searching.run import Run

    from convSearchPython.basics import conf
    from convSearchPython.dataset import Conversations, QueryMap
    from convSearchPython.utils import float_range_decimal
    from convSearchPython.utils.data_utils import guess_type, split_comma_not_quotes
    from convSearchPython.utils.imports import import_callable, instantiate_class
    from convSearchPython.utils.metrics import parse_metrics


    class CmdArgs(NamedTuple):
        props_file: Path
        proc_num: int

        def __str__(self):
            return f'(\n' \
                   f'  props_file = {self[0]}\n' \
                   f'  proc_num   = {self[1]}\n' \
                   f')'


    NoneCallableDatasetFn = Callable[[], Tuple[DataFrame, DataFrame, Conversations, QueryMap]]
    StrCallableDatasetFn = Callable[[str], Tuple[DataFrame, DataFrame, Conversations, QueryMap]]


    class DatasetSettingsProps(NamedTuple):
        fn: Union[NoneCallableDatasetFn, StrCallableDatasetFn]
        value: str


    class RunSettingsProps(NamedTuple):
        name: str
        trec: bool
        csv: bool
        pkl: bool
        limit: int


    class MeasureSettingsProps(NamedTuple):
        txt: bool
        csv: bool
        pkl: bool
        parsable: bool


    class SettingsProps(NamedTuple):
        dataset: DatasetSettingsProps
        include_text: bool
        index: str
        run: RunSettingsProps
        measure: MeasureSettingsProps
        metrics: List[Any]
        pipelines: Tuple[str]

        def __str__(self):
            return f'(\n' \
                   f'  dataset.fn            = {self.dataset.fn.__qualname__}\n' \
                   f'  dataset.value         = {self.dataset.value}\n' \
                   f'  include_text          = {self.include_text}\n' \
                   f'  index                 = {self.index}\n' \
                   f'  run.name              = {self.run.name}\n' \
                   f'  run.trec              = {self.run.trec}\n' \
                   f'  run.csv               = {self.run.csv}\n' \
                   f'  run.pkl               = {self.run.pkl}\n' \
                   f'  run.limit             = {self.run.limit}\n' \
                   f'  measures.txt          = {self.measure.txt}\n' \
                   f'  measures.csv          = {self.measure.csv}\n' \
                   f'  measures.pkl          = {self.measure.pkl}\n' \
                   f'  measures.parsable     = {self.measure.parsable}\n' \
                   f'  metrics               = {self.metrics}\n' \
                   f'  pipelines / factories = {self.pipelines}\n' \
                   f')'


    class ConfSection(Mapping[str, Tuple[Any, ...]]):
        def __init__(self, props: SectionProxy, rel_path: Path):
            self._map = _map = defaultdict(tuple)
            for key, value in props.items():
                if key.endswith('.$range'):
                    _map[key[:-7]] = _map[key[:-7]] + self.parse_range(value)
                elif key.endswith('.$file'):
                    file = Path(value)
                    if not file.is_absolute():
                        file = Path(rel_path, file)
                    if not file.exists():
                        raise IOError(f'File {file} do not exists')
                    file_conf = ConfigParser()
                    file_conf.read(str(file))
                    if 'RANGE' in file_conf:
                        for r_key, r_value in file_conf['RANGE'].items():
                            r_key = key[:-6] + '.' + r_key
                            _map[r_key] = _map[r_key] + self.parse_range(r_value)
                    if 'VALUE' in file_conf:
                        for v_key, v_value in file_conf['VALUE'].items():
                            v_key = key[:-6] + '.' + v_key
                            _map[v_key] = _map[v_key] + self.parse_list(v_value)
                else:
                    _map[key] = _map[key] + self.parse_list(value)

        @staticmethod
        def parse_range(value: str) -> Tuple[Union[int, float], ...]:
            parts = list(x.strip() for x in value.split(','))
            if len(parts) != 3:
                raise ValueError(f'Invalid range {parts}, range should be [start, stop, step]')
            start, stop, step = parts
            return tuple(guess_type(str(x), no_bool=True, no_str=True) for x in float_range_decimal(start, stop, step))

        @staticmethod
        def parse_list(value: str) -> Tuple[Any, ...]:
            return tuple(guess_type(x.strip(), allow_quote_for_str=True) for x in split_comma_not_quotes(value))

        def __getitem__(self, k: str) -> Tuple[Any, ...]:
            if k in self._map:
                return self._map[k]
            raise KeyError(k)

        def __len__(self) -> int:
            return len(self._map)

        def __iter__(self) -> Iterator[str]:
            return iter(self._map.keys())


    class Props(NamedTuple):
        settings: SettingsProps
        pipelines: ConfSection
        factories: ConfSection


    def cmd_parse() -> CmdArgs:
        """Parse command line arguments"""
        arg_parser = ArgumentParser(description='Main executable for convSearchPython')
        arg_parser.add_argument('PROPS', type=Path, help='Search properties (ini file)')
        arg_parser.add_argument('-p, --processes', type=int, dest='procs', default=1,
                                help='Number of worker processes (default to 1)')
        opt = arg_parser.parse_args()
        if not opt.PROPS.exists():
            raise IOError(f'Props file "{opt.PROPS}" do not exists')
        if opt.procs <= 0:
            raise ValueError(f'The number of process should be greater than 0')
        return CmdArgs(opt.PROPS, opt.procs)


    def load_props(file: Path):
        """Load props file"""
        props = ConfigParser()
        props.read(str(file))
        settings = 'SETTINGS'

        return Props(
            SettingsProps(
                DatasetSettingsProps(
                    import_callable(props.get(settings, 'dataset.fn')),
                    props.get(settings, 'dataset.value', fallback=None)
                ),
                props.getboolean(settings, 'include.text', fallback=False),
                props.get(settings, 'index'),
                RunSettingsProps(
                    props.get(settings, 'run.name', fallback='test'),
                    props.getboolean(settings, 'run.trec', fallback=True),
                    props.getboolean(settings, 'run.csv', fallback=False),
                    props.getboolean(settings, 'run.pkl', fallback=False),
                    props.getint(settings, 'run.limit', fallback=-1)
                ),
                MeasureSettingsProps(
                    props.getboolean(settings, 'measures.txt', fallback=True),
                    props.getboolean(settings, 'measures.csv', fallback=False),
                    props.getboolean(settings, 'measures.pkl', fallback=False),
                    props.getboolean(settings, 'measures.parsable', fallback=True)
                ),
                parse_metrics(props.get(settings, 'metrics', fallback='AP, nDCG@3, P@1, P@3, RR, R@200, R@100')),
                tuple(x.strip() for x in props.get(settings, 'pipelines').split(','))
            ),
            ConfSection(props['PIPELINES'] if 'PIPELINES' in props else {}, file.parent),
            ConfSection(props['FACTORIES'] if 'FACTORIES' in props else {}, file.parent)
        )


    def load_common_objects(props: Props) -> Dict[str, Any]:
        """Load common objects"""
        if props.settings.dataset.value is not None:
            tpl = props.settings.dataset.fn(props.settings.dataset.value)
        else:
            tpl = props.settings.dataset.fn()

        cache_dir = conf.get('GENERAL', 'cache_dir', fallback=None)
        if cache_dir is not None:
            cache_dir = Path(cache_dir).absolute()

        return {
            'index': props.settings.index,
            'queries': tpl[0],
            'qrels': tpl[1],
            'conversations': tpl[2],
            'query_map': tpl[3],
            'metadata': ['docno', 'text'] if props.settings.include_text else ['docno'],
            'cache_dir': cache_dir,
        }


    def unroll_variants(values: Dict[str, Any]) -> List[Dict[str, Any]]:
        rolled = {k: v for k, v in values.items() if isinstance(v, tuple) or isinstance(v, list)}
        values = {k: v for k, v in values.items() if k not in rolled}

        unrolled = []
        r_keys = list(rolled.keys())
        r_values = [rolled[k] for k in r_keys]
        for prod in itertools.product(*r_values):
            v = {r_keys[i]: prod[i] for i in range(len(r_keys))}
            v.update(values)
            unrolled.append(v)
        return unrolled


    def load_pipelines(props: Props, common_objs: Dict[str, Any]) -> List[Pipeline]:
        """Load pipelines"""
        pipelines = []

        for name in props.settings.pipelines:
            if name in props.pipelines and name in props.factories:
                raise ValueError(f'pipeline name "{name}" defined in both pipelines and factories. '
                                 f'Make sure names are unique.')
            if name in props.pipelines:
                is_factory = False
                values = {k: v for k, v in props.pipelines.items() if k == name or k.startswith('{}.'.format(name))}
            elif name in props.factories:
                is_factory = True
                values = {k: v for k, v in props.factories.items() if k == name or k.startswith('{}.'.format(name))}
            else:
                raise ValueError(f'pipeline name "{name}" is not defined in props file')

            classname = values.pop(name)
            if classname is None:
                raise ValueError(f'missing classname for pipeline "{name}"')
            values = {k[len(name)+1:]: v for k, v in values.items()}
            values.pop('', None)
            variants = unroll_variants(values)

            if not is_factory:
                for var in variants:
                    print(f'Instantiating class <{classname[0]}> with args {var}', flush=True)
                    pipelines.append(instantiate_class(classname[0], **{**common_objs, **var}))
            else:
                for var in variants:
                    print(f'Building using factory <{classname[0]}> with args {var}')
                    factory: PipelineFactory = instantiate_class(classname[0], **common_objs)
                    factory.set(var)
                    pipelines.append(factory.build())

        return pipelines


    def run_pipelines(props: Props, pipelines: List[Pipeline], pool, start_time, common_objects: Dict[str, Any]):
        """Run pipelines"""
        run = Run(name=props.settings.run.name, parallel_pool=pool)
        for p in pipelines:
            run.add(p, discard_dup=True)

        print(f'Start execution (elapsed {timedelta(seconds=(time() - start_time))})', flush=True)
        run.execute(common_objects['queries'], common_objects.get('qrels'), props.settings.run.limit)

        print(f'Saving run (elapsed {timedelta(seconds=(time() - start_time))})', flush=True)
        if props.settings.run.trec:
            run.save_as_trec()
        if props.settings.run.csv:
            run.save_as_csv()
        if props.settings.run.pkl:
            run.save_as_pkl()

        print(f'Calculating measures (elapsed {timedelta(seconds=(time() - start_time))})', flush=True)
        measures_dict = run.get_measures(props.settings.metrics, common_objects['qrels'], common_objects['query_map'])

        print(f'Saving measures (elapsed {timedelta(seconds=(time() - start_time))})', flush=True)
        for key, measure in measures_dict.items():
            if key.startswith('parsable_'):
                if props.settings.measure.parsable:
                    measure.save_as_csv()
            else:
                if props.settings.measure.txt:
                    measure.save_simple()
                if props.settings.measure.csv:
                    measure.save_as_csv()
                if props.settings.measure.pkl:
                    measure.save_as_pkl()

        print(f'Measures saved (elapsed {timedelta(seconds=(time() - start_time))})', flush=True)


    def main():
        print('Started convSearchPython', flush=True)
        start_time = time()
        cmd_args = cmd_parse()
        props = load_props(cmd_args.props_file)
        print(f'Loaded run {props.settings.run.name} with settings:',
              props.settings,
              sep='\n',
              end='\n\n',
              flush=True)
        common_objs = load_common_objects(props)
        pipelines = load_pipelines(props, common_objs)
        mp.set_start_method('spawn', force=True)
        if cmd_args.proc_num > 1:
            print(f'Starting a parallel pool of {cmd_args.proc_num} workers', flush=True)
            with mp.get_context('spawn').Pool(cmd_args.proc_num, initializer=_proc_init) as pool:
                run_pipelines(props, pipelines, pool, start_time, common_objs)
        else:
            run_pipelines(props, pipelines, None, start_time, common_objs)
        print(f'Total elapsed time: {timedelta(seconds=(time() - start_time))}',
              'END', sep='\n', flush=True)


    main()
