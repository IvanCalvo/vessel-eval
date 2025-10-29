from pyvane import pipeline
from functools import partial
from utils.gt_seg import Gt_seg

channel = None
name_filter = None
save_steps = 'all'
start_at = 0
verbosity = 3

img_reader = partial(pipeline.read_and_adjust_img, channel=channel)

def convert_to_graph(input_path, output_path):
    dp = pipeline.BasePipeline(input_path, img_reader, output_path=output_path, 
                            name_filter=name_filter, save_steps=save_steps, start_at=start_at, verbosity=verbosity)

    segmenter = Gt_seg()
    skeleton_builder = pipeline.DefaultSkeletonBuilder()
    network_builder = pipeline.DefaultNetworkBuilder()
    analyzer = pipeline.DefaultAnalyzer(10)

    dp.set_processors(segmenter, skeleton_builder, network_builder)
    graph_target = dp._run_one_file(dp.files[0])
    graph_scores = dp._run_one_file(dp.files[1])

    return graph_target, graph_scores