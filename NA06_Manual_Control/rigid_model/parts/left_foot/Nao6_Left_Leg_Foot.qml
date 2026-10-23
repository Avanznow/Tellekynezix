import QtQuick
import QtQuick3D

Node {
    id: node

    // Resources
    PrincipledMaterial {
        id: plastico_BLANCO_010_material
        objectName: "PLASTICO_BLANCO.010"
        baseColor: "#ffcccccc"
        indexOfRefraction: 1.4500000476837158
    }
    PrincipledMaterial {
        id: metalico_naranja_010_material
        objectName: "Metalico_naranja.010"
        baseColor: "#ffcc1104"
        indexOfRefraction: 1.4500000476837158
    }

    // Nodes:
    Node {
        id: nao6_Left_Leg_Foot_obj
        objectName: "Nao6_Left_Leg_Foot.obj"
        Model {
            id: nao6_Left_Leg_Foot
            objectName: "Nao6_Left_Leg_Foot"
            source: "meshes/nao6_Left_Leg_Foot_mesh.mesh"
            materials: [
                plastico_BLANCO_010_material,
                metalico_naranja_010_material
            ]
        }
    }

    // Animations:
}
