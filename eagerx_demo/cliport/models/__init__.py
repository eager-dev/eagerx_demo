from eagerx_demo.cliport.models.resnet import ResNet43_8s, DummyResNet43_8s
from eagerx_demo.cliport.models.resnet_lat import ResNet45_10s, DummyResNet45_10s
from eagerx_demo.cliport.models.clip_lingunet_lat import CLIPLingUNetLat, DummyCLIPLingUNetLat


names = {
    # resnet
    "plain_resnet": ResNet43_8s,
    "dummy_plain_resnet": DummyResNet43_8s,
    # lateral connections
    "plain_resnet_lat": ResNet45_10s,
    "dummy_plain_resnet_lat": DummyResNet45_10s,
    "clip_lingunet_lat": CLIPLingUNetLat,
    "dummy_clip_lingunet_lat": DummyCLIPLingUNetLat,
}
