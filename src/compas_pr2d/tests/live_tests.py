import interaction.contact_2d as contact
import compas.geometry as cg
from arch import construct_arch
from problem.variables import _remap_polygons

Arch_Object = construct_arch(5.0, 20.0, 10, 1.0)
Arch_Object = _remap_polygons(Arch_Object)
Interfaces = contact.contact_2d_edge(Arch_Object)
print("Detected Interfaces:", Interfaces)
