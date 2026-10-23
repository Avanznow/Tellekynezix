import QtQuick
import QtQuick3D

Node {
    id: node

    // Resources
    PrincipledMaterial {
        id: plastico_BLANCO_015_material
        objectName: "PLASTICO_BLANCO.015"
        baseColor: "#ffcccccc"
        indexOfRefraction: 1.4500000476837158
    }
    PrincipledMaterial {
        id: metalico_naranja_015_material
        objectName: "Metalico_naranja.015"
        baseColor: "#ffcc1104"
        indexOfRefraction: 1.4500000476837158
    }

    // Nodes:
    Node {
        id: nao6_Right_Arm_Forearm_obj
        objectName: "Nao6_Right_Arm_Forearm.obj"
        Model {
            id: nao6_Right_Arm_Forearm
            objectName: "Nao6_Right_Arm_Forearm"
            source: "meshes/nao6_Right_Arm_Forearm_mesh.mesh"
            materials: [
                plastico_BLANCO_015_material,
                metalico_naranja_015_material
            ]
        }
    }

    // Animations:
}
