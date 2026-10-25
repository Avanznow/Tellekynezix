import QtQuick
import QtQuick3D

Node {
    id: node

    // Resources
    PrincipledMaterial {
        id: plastico_BLANCO_001_material
        objectName: "PLASTICO_BLANCO.001"
        baseColor: "#ffcccccc"
        indexOfRefraction: 1.4500000476837158
    }

    // Nodes:
    Node {
        id: nao6_Left_Arm_Bicep_obj
        objectName: "Nao6_Left_Arm_Bicep.obj"
        Model {
            id: nao6_Left_Arm_Bicep
            objectName: "Nao6_Left_Arm_Bicep"
            source: "meshes/nao6_Left_Arm_Bicep_mesh.mesh"
            materials: [
                plastico_BLANCO_001_material
            ]
        }
    }

    // Animations:
}
